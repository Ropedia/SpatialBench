from typing import Any, Dict

import torch
import torch.nn.functional as F


def summarize_online_frame_feat(feat: torch.Tensor | None):
    if feat is None or not isinstance(feat, torch.Tensor):
        return None

    feat = feat.detach().float()
    if feat.dim() == 1:
        return F.normalize(feat, dim=0, eps=1e-8)

    if feat.dim() >= 2 and feat.shape[1] == 1:
        feat = feat[:, 0]

    if feat.dim() == 4:
        feat = feat.reshape(feat.shape[0] * feat.shape[1], feat.shape[2], feat.shape[3])

    if feat.dim() == 3:
        feat = F.normalize(feat, dim=-1, eps=1e-8)
        feat = feat.mean(dim=1)
    elif feat.dim() == 2:
        feat = F.normalize(feat, dim=-1, eps=1e-8)
    else:
        feat = feat.reshape(feat.shape[0], -1)
        feat = F.normalize(feat, dim=-1, eps=1e-8)

    feat = feat.mean(dim=0)
    return F.normalize(feat, dim=0, eps=1e-8)


def get_online_similarity_cache_key(frame_id_a: int, frame_id_b: int):
    return (
        (frame_id_a, frame_id_b)
        if frame_id_a <= frame_id_b
        else (frame_id_b, frame_id_a)
    )


def log_online_memory_selection(
    info: Dict[str, Any],
    similarity_threshold: float,
):
    current_frame_id = info.get("current_frame_id", None)
    keep_ids = info.get("keep_ids", [])
    recent_ids = info.get("recent_ids", [])
    selected_memory_ids = info.get("selected_memory_ids", [])
    candidate_logs = info.get("candidate_logs", [])

    print(
        "[R3][online-memory] "
        f"current={current_frame_id} keep={keep_ids} recent={recent_ids} "
        f"selected_memory={selected_memory_ids} threshold={similarity_threshold:.3f}"
    )
    for candidate in candidate_logs:
        frame_id = candidate["frame_id"]
        confidence = candidate["confidence"]
        max_similarity = candidate["max_similarity"]
        novelty = candidate["novelty"]
        utility = candidate["utility"]
        decision = candidate["decision"]
        reason = candidate.get("reason", "")
        compared_ids = candidate.get("compared_ids", [])
        print(
            "[R3][online-memory] "
            f"candidate={frame_id} conf={confidence:.4f} "
            f"max_sim={max_similarity:.4f} novelty={novelty:.4f} "
            f"utility={utility:.4f} compared={compared_ids} "
            f"decision={decision} {reason}".rstrip()
        )
