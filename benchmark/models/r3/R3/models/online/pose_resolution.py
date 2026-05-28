import torch

from R3.utils.pose_utils import reconstruct_camera_sequence_from_rel_pose, refine_camera_sequence_from_rel_pose


def resolve_reconstructed_pose_enc(
    predictions,
    rel_pose_enc: torch.Tensor,
    rel_pose_conf: torch.Tensor,
    rel_pose_mask: torch.Tensor,
    rel_pose_reconstruction_method: str,
    rel_pose_reconstruction_kwargs: dict | None,
    *,
    rel_pose_conf_t: torch.Tensor | None = None,
    rel_pose_conf_r: torch.Tensor | None = None,
    online_mode: bool = False,
    rel_pose_frame_ids=None,
    anchored_pose_enc: torch.Tensor | None = None,
):
    # Resolve absolute pose encodings from relative pose predictions.
    rel_pose_reconstruction_kwargs = dict(rel_pose_reconstruction_kwargs or {})
    rel_pose_reconstruction_kwargs.setdefault("topn_conf", 10)

    def reconstruct_pred_poses():
        return reconstruct_camera_sequence_from_rel_pose(
            rel_pose_enc,
            rel_pose_conf,
            rel_pose_mask,
            pred_rel_conf_t=rel_pose_conf_t,
            pred_rel_conf_r=rel_pose_conf_r,
            method=rel_pose_reconstruction_method,
            **rel_pose_reconstruction_kwargs,
        )

    if not online_mode:
        return reconstruct_pred_poses()

    if rel_pose_frame_ids is not None:
        predictions["rel_pose_frame_ids"] = rel_pose_frame_ids
    if anchored_pose_enc is not None:
        return anchored_pose_enc
    if rel_pose_frame_ids is None:
        return reconstruct_pred_poses()
    if rel_pose_enc.shape[1] <= 1:
        pred_poses = reconstruct_pred_poses()
        predictions["pose_enc_pool_local"] = pred_poses
        return pred_poses[:, -1:]

    pred_poses = reconstruct_pred_poses()
    predictions["pose_enc_pool_local"] = pred_poses
    return pred_poses[:, -1:]


def build_online_rel_pose_graph(merged_predictions, *, output_frame_ids=None):
    # Collapse per-step online relative-pose predictions into one output-frame graph.
    rel_pose_enc_list = merged_predictions.get("rel_pose_enc_list")
    rel_pose_conf_list = merged_predictions.get("rel_pose_conf_list")
    rel_pose_mask_list = merged_predictions.get("rel_pose_mask_list")
    rel_pose_frame_ids_list = merged_predictions.get("rel_pose_frame_ids_list")
    rel_pose_conf_t_list = merged_predictions.get("rel_pose_conf_t_list")
    rel_pose_conf_r_list = merged_predictions.get("rel_pose_conf_r_list")
    pose_enc = merged_predictions.get("pose_enc")
    if not (
        isinstance(rel_pose_enc_list, list)
        and isinstance(rel_pose_conf_list, list)
        and isinstance(rel_pose_mask_list, list)
        and isinstance(pose_enc, torch.Tensor)
    ):
        return None

    if output_frame_ids is None:
        output_frame_ids = list(range(int(pose_enc.shape[1])))
    else:
        output_frame_ids = [int(frame_id) for frame_id in output_frame_ids]

    if len(output_frame_ids) != pose_enc.shape[1] or not output_frame_ids:
        return None

    example_rel_pose = next((item for item in rel_pose_enc_list if isinstance(item, torch.Tensor)), None)
    if example_rel_pose is None:
        return None

    batch_size = example_rel_pose.shape[0]
    seq_len = len(output_frame_ids)
    device = example_rel_pose.device
    dtype = example_rel_pose.dtype
    output_index_by_frame_id = {frame_id: idx for idx, frame_id in enumerate(output_frame_ids)}

    rel_pose_enc = torch.zeros(batch_size, seq_len, seq_len, 9, device=device, dtype=dtype)
    rel_pose_conf = torch.full((batch_size, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    rel_pose_mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)

    has_split_confidence = isinstance(rel_pose_conf_t_list, list) and isinstance(rel_pose_conf_r_list, list)
    rel_pose_conf_t = None
    rel_pose_conf_r = None
    if has_split_confidence:
        rel_pose_conf_t = torch.full((batch_size, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        rel_pose_conf_r = torch.full((batch_size, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)

    for step_idx, step_rel_pose_enc in enumerate(rel_pose_enc_list):
        step_rel_pose_conf = rel_pose_conf_list[step_idx] if step_idx < len(rel_pose_conf_list) else None
        step_rel_pose_mask = rel_pose_mask_list[step_idx] if step_idx < len(rel_pose_mask_list) else None
        if not (
            isinstance(step_rel_pose_enc, torch.Tensor)
            and isinstance(step_rel_pose_conf, torch.Tensor)
            and isinstance(step_rel_pose_mask, torch.Tensor)
        ):
            continue

        if rel_pose_frame_ids_list is not None and step_idx < len(rel_pose_frame_ids_list):
            frame_ids = [int(frame_id) for frame_id in rel_pose_frame_ids_list[step_idx]]
        else:
            frame_ids = list(range(step_rel_pose_enc.shape[1]))

        if len(frame_ids) != step_rel_pose_enc.shape[1]:
            continue

        step_rel_pose_conf_t = None
        step_rel_pose_conf_r = None
        if has_split_confidence:
            if step_idx < len(rel_pose_conf_t_list) and isinstance(rel_pose_conf_t_list[step_idx], torch.Tensor):
                step_rel_pose_conf_t = rel_pose_conf_t_list[step_idx]
            if step_idx < len(rel_pose_conf_r_list) and isinstance(rel_pose_conf_r_list[step_idx], torch.Tensor):
                step_rel_pose_conf_r = rel_pose_conf_r_list[step_idx]

        for local_i, global_i in enumerate(frame_ids):
            output_i = output_index_by_frame_id.get(global_i)
            if output_i is None:
                continue

            for local_j, global_j in enumerate(frame_ids):
                output_j = output_index_by_frame_id.get(global_j)
                if output_j is None:
                    continue

                update_mask = step_rel_pose_mask[:, local_i, local_j].bool()
                if not update_mask.any():
                    continue

                better_mask = update_mask & (
                    step_rel_pose_conf[:, local_i, local_j] > rel_pose_conf[:, output_i, output_j]
                )
                if not better_mask.any():
                    continue

                rel_pose_conf[:, output_i, output_j] = torch.where(
                    better_mask,
                    step_rel_pose_conf[:, local_i, local_j],
                    rel_pose_conf[:, output_i, output_j],
                )
                rel_pose_mask[:, output_i, output_j] = rel_pose_mask[:, output_i, output_j] | better_mask
                rel_pose_enc[:, output_i, output_j] = torch.where(
                    better_mask.view(-1, 1),
                    step_rel_pose_enc[:, local_i, local_j],
                    rel_pose_enc[:, output_i, output_j],
                )

                if rel_pose_conf_t is not None and rel_pose_conf_r is not None:
                    source_conf_t = step_rel_pose_conf_t
                    source_conf_r = step_rel_pose_conf_r
                    if not isinstance(source_conf_t, torch.Tensor) or not isinstance(source_conf_r, torch.Tensor):
                        source_conf_t = step_rel_pose_conf
                        source_conf_r = step_rel_pose_conf
                    rel_pose_conf_t[:, output_i, output_j] = torch.where(
                        better_mask,
                        source_conf_t[:, local_i, local_j],
                        rel_pose_conf_t[:, output_i, output_j],
                    )
                    rel_pose_conf_r[:, output_i, output_j] = torch.where(
                        better_mask,
                        source_conf_r[:, local_i, local_j],
                        rel_pose_conf_r[:, output_i, output_j],
                    )

    return rel_pose_enc, rel_pose_conf, rel_pose_mask, rel_pose_conf_t, rel_pose_conf_r


def finalize_online_pose_sequence(
    merged_predictions,
    rel_pose_reconstruction_method: str,
    rel_pose_reconstruction_kwargs: dict | None,
    *,
    output_frame_ids=None,
):
    # Finalize one online output chunk with a global reconstruction pass over merged edges.
    pose_enc = merged_predictions.get("pose_enc")
    if not isinstance(pose_enc, torch.Tensor) or pose_enc.shape[1] <= 1:
        return None

    rel_pose_graph = build_online_rel_pose_graph(merged_predictions, output_frame_ids=output_frame_ids)
    if rel_pose_graph is None:
        return None

    rel_pose_enc, rel_pose_conf, rel_pose_mask, rel_pose_conf_t, rel_pose_conf_r = rel_pose_graph
    if not rel_pose_mask.any():
        return None

    rel_pose_reconstruction_kwargs = dict(rel_pose_reconstruction_kwargs or {})
    rel_pose_reconstruction_kwargs.setdefault("topn_conf", 10)

    if rel_pose_reconstruction_method == "pgo":
        return refine_camera_sequence_from_rel_pose(
            rel_pose_enc,
            rel_pose_conf,
            rel_pose_mask,
            init_pose_enc=pose_enc,
            pred_rel_conf_t=rel_pose_conf_t,
            pred_rel_conf_r=rel_pose_conf_r,
            method=rel_pose_reconstruction_method,
            **rel_pose_reconstruction_kwargs,
        )

    return reconstruct_camera_sequence_from_rel_pose(
        rel_pose_enc,
        rel_pose_conf,
        rel_pose_mask,
        pred_rel_conf_t=rel_pose_conf_t,
        pred_rel_conf_r=rel_pose_conf_r,
        method=rel_pose_reconstruction_method,
        **rel_pose_reconstruction_kwargs,
    )
