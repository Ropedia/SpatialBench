from dataclasses import dataclass, field


@dataclass(slots=True)
class OnlineState:
    batch_size: int
    frame_count: int = 0
    last_fallback_frame_id: int = -1
    pending_fallback_action: object = None
    kv_cache_list: object = None
    paged_kv_store: object = None
    tokens_per_frame: object = None
    cache_frame_ids: list = field(default_factory=list)
    frame_order: list = field(default_factory=list)
    memory_bank_ids: list = field(default_factory=list)
    segment_anchor_frame_ids: list = field(default_factory=list)
    frame_feats: dict = field(default_factory=dict)
    frame_rel_pose_feats: dict = field(default_factory=dict)
    frame_select_feats: dict = field(default_factory=dict)
    frame_pose_enc: dict = field(default_factory=dict)
    frame_scores: dict = field(default_factory=dict)
    frame_post_scores: dict = field(default_factory=dict)
    frame_score_history: dict = field(default_factory=dict)
    similarity_cache: dict = field(default_factory=dict)
    scale_factor: float = 1.0

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)
