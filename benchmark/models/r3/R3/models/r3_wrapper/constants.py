"""Shared constants for the R3 wrapper and runtime mixins."""

ONLINE_WRAPPER_OPTION_KEYS = {
    "export_feat_layers",
    "rel_pose_reconstruction_method",
    "rel_pose_reconstruction_kwargs",
    "online_verbose",
    "online_memory_verbose",
    # Legacy keys kept in the filter to prevent them leaking into da3_kwargs
    "online_finalize_pose_reconstruction",
    "online_revisit_enabled",
    "online_revisit_max_iterations",
    "online_revisit_min_improvement",
    "online_revisit_min_confidence",
    "online_revisit_verbose",
    "runtime_stats_every",
    "runtime_stats_path",
}
