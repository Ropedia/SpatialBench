import json
import time
from typing import Dict, Optional

import torch


def resolve_online_verbose(
    online_verbose: Optional[bool],
    online_memory_verbose: Optional[bool],
    online_revisit_verbose: Optional[bool],
):
    # Resolve the unified online verbose flag while keeping backward-compatible aliases.
    if online_verbose is not None:
        return bool(online_verbose)
    if online_memory_verbose is not None:
        return bool(online_memory_verbose)
    if online_revisit_verbose is not None:
        return bool(online_revisit_verbose)
    return True


def is_online_verbose(wrapper, online_options: Optional[Dict] = None):
    # Return whether online diagnostic logging is enabled for the current call.
    if online_options is None:
        return wrapper.online_verbose
    return bool(online_options.get("online_verbose", wrapper.online_verbose))


def order_online_predictions_by_frame_ids(merged_predictions, processed_frame_ids, output_frame_ids):
    # Restore per-frame tensors to the caller's original frame order after replay.
    if processed_frame_ids == output_frame_ids:
        return merged_predictions

    frame_position = {frame_id: idx for idx, frame_id in enumerate(processed_frame_ids)}
    ordered_indices = [frame_position[frame_id] for frame_id in output_frame_ids]
    index_tensor = None
    for key in {
        "depth",
        "depth_conf",
        "pose_enc",
        "images",
        "sky",
        "sky_mask",
        "non_sky_mask",
    }:
        value = merged_predictions.get(key)
        if isinstance(value, torch.Tensor) and value.shape[1] == len(processed_frame_ids):
            if index_tensor is None:
                index_tensor = torch.tensor(ordered_indices, device=value.device, dtype=torch.long)
            merged_predictions[key] = value.index_select(1, index_tensor)

    pose_enc_pool_local = merged_predictions.get("pose_enc_pool_local")
    if isinstance(pose_enc_pool_local, list) and len(pose_enc_pool_local) == len(processed_frame_ids):
        merged_predictions["pose_enc_pool_local"] = [pose_enc_pool_local[idx] for idx in ordered_indices]
    return merged_predictions


def run_online_sequence_pass(
    wrapper,
    images,
    online_options,
    da3_kwargs,
    base_state=None,
    frame_ids=None,
    output_frame_ids=None,
    pose_max_recent=0,
):
    # Replay a sequence chunk against a cloned online state and return canonicalized outputs.
    if images.dim() == 4:
        images = images.unsqueeze(1)

    batch_size, seq_len = images.shape[:2]
    if base_state is None:
        state = wrapper._create_online_state(batch_size)
    else:
        if base_state.batch_size != batch_size:
            raise ValueError("base_state batch size does not match the replay batch size")
        state = wrapper._clone_online_state(base_state)

    if frame_ids is None:
        start_frame_id = int(state.frame_count)
        frame_ids = list(range(start_frame_id, start_frame_id + seq_len))
    else:
        frame_ids = list(frame_ids)
    if output_frame_ids is None:
        output_frame_ids = sorted(frame_ids)

    predictions_list = []
    processed_frame_ids = []
    stats_every = int(online_options.get("runtime_stats_every", 0) or 0)
    stats_path = online_options.get("runtime_stats_path") or ""
    stats_start = time.time()
    for step_idx, frame_id in enumerate(frame_ids):
        frame_images = images[:, step_idx : step_idx + 1]
        predictions = wrapper._run_online_step(
            frame_images,
            state,
            online_options=online_options,
            da3_kwargs=da3_kwargs,
            current_frame_id=frame_id,
            pose_max_recent=pose_max_recent,
        )
        pending_fallback_action = state.pending_fallback_action
        state.pending_fallback_action = None
        if pending_fallback_action is not None:
            wrapper._execute_fallback(pending_fallback_action, state, online_options, da3_kwargs)
        if frame_id in getattr(state, "frame_order", []):
            predictions_list.append(predictions)
            processed_frame_ids.append(frame_id)
        if stats_every > 0 and stats_path and (
            (step_idx + 1) % stats_every == 0 or step_idx + 1 == len(frame_ids)
        ):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            else:
                vram_gb = 0.0
            elapsed = max(time.time() - stats_start, 1e-6)
            record = {
                "frame": int(frame_id),
                "fps": float((step_idx + 1) / elapsed),
                "vram_gb": float(vram_gb),
            }
            with open(stats_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    effective_output_frame_ids = [
        frame_id for frame_id in output_frame_ids if frame_id in processed_frame_ids
    ]
    if not predictions_list:
        raise RuntimeError("Online pass produced no retained frame predictions")
    if len(predictions_list) == 1:
        merged_predictions = predictions_list[0]
    else:
        merged_predictions = wrapper._merge_online_step_predictions(predictions_list)
        merged_predictions = order_online_predictions_by_frame_ids(
            merged_predictions,
            processed_frame_ids,
            effective_output_frame_ids,
        )

    # Prefer wrapper._persistent_post_scores (written during each normal-inference
    # step, survives fallback flush). Fall back to the state dict for frames that
    # only ever appeared via replay.
    persistent_post_scores = getattr(wrapper, "_persistent_post_scores", {}) or {}
    merged_predictions["frame_post_scores"] = {
        frame_id: float(
            persistent_post_scores.get(
                frame_id,
                state.frame_post_scores.get(frame_id, float("inf")),
            )
        )
        for frame_id in effective_output_frame_ids
    }
    merged_predictions["frame_score_history"] = {
        frame_id: list(state.frame_score_history.get(frame_id, []))
        for frame_id in effective_output_frame_ids
    }
    merged_predictions["output_frame_ids"] = list(effective_output_frame_ids)
    return merged_predictions, state


def get_online_revisit_candidate(frame_ids, frame_post_scores, min_confidence):
    # Pick the lowest-scored frame that still falls below the revisit threshold.
    candidate_id = None
    candidate_score = None
    for frame_id in frame_ids:
        frame_score = float(frame_post_scores.get(frame_id, float("inf")))
        if frame_score >= min_confidence:
            continue
        if candidate_id is None or frame_score < candidate_score:
            candidate_id = frame_id
            candidate_score = frame_score
    return candidate_id, candidate_score


def build_online_revisit_frame_order(frame_ids, revisit_frame_id):
    # Move the selected revisit frame to the end while preserving all other order.
    if revisit_frame_id not in frame_ids:
        raise ValueError(f"Unknown revisit frame id: {revisit_frame_id}")
    return [frame_id for frame_id in frame_ids if frame_id != revisit_frame_id] + [revisit_frame_id]


def log_online_revisit(wrapper, online_options, message: str):
    # Emit concise revisit diagnostics when verbose mode is enabled.
    if is_online_verbose(wrapper, online_options):
        print(f"[R3][online-revisit] {message}")


def has_sufficient_revisit_improvement(candidate_score: float, replay_score: float, min_improvement: float):
    # Require strict improvement when the configured threshold is disabled.
    improvement = replay_score - candidate_score
    if min_improvement <= 0.0:
        return improvement > 0.0
    return improvement >= min_improvement


def get_online_revisit_repeated_score_stop_message(candidate_id: int, replay_score: float, revisit_score_map):
    # Stop revisit loops when replaying a candidate would reproduce the same accepted score.
    previous_score = revisit_score_map.get(candidate_id)
    if previous_score is None:
        return None
    if replay_score != previous_score:
        return None
    return f"stop: frame {candidate_id} replayed with unchanged score {replay_score:.4f}"


def run_online_revisit_loop(
    wrapper, images, base_state, frame_ids, merged_predictions, state, online_options, da3_kwargs
):
    # Revisit the current lowest-confidence frame by replaying the chunk with that frame moved to the end.
    if not online_options["online_revisit_enabled"]:
        return merged_predictions, state
    if wrapper.online_kv_cache_mode != "all":
        log_online_revisit(wrapper, online_options, "skip: revisit is only supported in online_kv_cache_mode='all'")
        return merged_predictions, state
    if len(frame_ids) <= 1:
        return merged_predictions, state
    if base_state is not None and int(base_state.frame_count) != 0:
        log_online_revisit(wrapper, online_options, "skip: revisit requires a fresh sequence state in v1")
        return merged_predictions, state

    min_confidence = online_options["online_revisit_min_confidence"]
    min_improvement = online_options["online_revisit_min_improvement"]
    max_iterations = online_options["online_revisit_max_iterations"]
    revisit_score_map = {}
    working_predictions = merged_predictions
    working_state = state

    for iteration_idx in range(max_iterations):
        frame_post_scores = working_state.frame_post_scores
        candidate_id, candidate_score = get_online_revisit_candidate(
            frame_ids,
            frame_post_scores,
            min_confidence,
        )
        if candidate_id is None:
            log_online_revisit(
                wrapper,
                online_options,
                f"stop: all frame scores reached the threshold {min_confidence:.3f}",
            )
            break

        replay_frame_ids = build_online_revisit_frame_order(frame_ids, candidate_id)
        replay_indices = torch.tensor(
            [frame_ids.index(frame_id) for frame_id in replay_frame_ids],
            device=images.device,
            dtype=torch.long,
        )
        replay_images = images.index_select(1, replay_indices)
        replay_predictions, replay_state = run_online_sequence_pass(
            wrapper,
            replay_images,
            online_options=online_options,
            da3_kwargs=da3_kwargs,
            base_state=base_state,
            frame_ids=replay_frame_ids,
            output_frame_ids=frame_ids,
        )
        replay_score = float(replay_state.frame_post_scores.get(candidate_id, float("-inf")))
        log_online_revisit(
            wrapper,
            online_options,
            (
                f"iter={iteration_idx} frame={candidate_id} order={replay_frame_ids} "
                f"old_score={candidate_score:.4f} new_score={replay_score:.4f}"
            ),
        )
        repeated_score_stop_message = get_online_revisit_repeated_score_stop_message(
            candidate_id,
            replay_score,
            revisit_score_map,
        )
        if repeated_score_stop_message is not None:
            log_online_revisit(wrapper, online_options, repeated_score_stop_message)
            break
        if not has_sufficient_revisit_improvement(candidate_score, replay_score, min_improvement):
            if min_improvement <= 0.0:
                message = f"stop: frame {candidate_id} did not improve"
            else:
                message = f"stop: frame {candidate_id} did not improve by at least {min_improvement:.4f}"
            log_online_revisit(wrapper, online_options, message)
            break

        working_predictions = replay_predictions
        working_state = replay_state
        revisit_score_map[candidate_id] = replay_score

    return working_predictions, working_state
