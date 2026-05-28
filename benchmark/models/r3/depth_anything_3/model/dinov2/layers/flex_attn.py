import torch
import torch._dynamo
from functools import lru_cache
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

flex_attention_compiled = torch.compile(flex_attention)
create_block_mask_compiled = torch.compile(create_block_mask)

# Allow enough compiled variants for all (S, resolution) combinations during training.
# Default limit (8-64) would trigger recompilation warnings with dynamic sequence lengths.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)


@lru_cache(maxsize=32)
def create_block_mask_cached(mask_mod, B, H, M, N, device):
    return create_block_mask_compiled(mask_mod, B, H, M, N, device=device)


def custom_mask_mod_with_params(
    block_size: int,
    look_forward: int,
    look_backward: int,
    sink_window: bool,
    dropped_frame_pairs=None,
):
    """
    Factory function to create a mask_mod callable with specific windowing parameters.

    Args:
        block_size (int): The size of each attention window.
        look_forward (int): Number of subsequent windows a query can attend to.
        look_backward (int): Number of preceding windows a query can attend to.
        sink_window (bool): If True, all queries can also attend to the first window of keys.

    Returns:
        _mask_mod_signature: A callable that conforms to the mask_mod signature,
                             taking (b, h, q_idx, kv_idx) and returning a boolean mask tensor.
    """
    dropped_frame_pairs_tensor = None
    if dropped_frame_pairs is not None:
        if isinstance(dropped_frame_pairs, torch.Tensor):
            dropped_frame_pairs_tensor = dropped_frame_pairs
        else:
            dropped_frame_pairs_tensor = torch.as_tensor(dropped_frame_pairs, dtype=torch.bool)

    def mask_mod(
        b: int,  # Batch index (often unused if mask is same across batches)
        h: int,  # Head index (often unused if mask is same across heads)
        q_idx: torch.Tensor,  # Tensor of query indices
        kv_idx: torch.Tensor,  # Tensor of key/value indices
    ):
        """
        Generates a boolean mask based on windowed attention with optional sink.
        """

        # Determine the window index for each query and key/value index
        q_window_idx = (q_idx) // block_size
        kv_window_idx = (kv_idx) // block_size

        local_min_kv_window = q_window_idx - look_backward
        local_max_kv_window = q_window_idx + look_forward

        # Boolean tensor indicating if kv_window_idx is within the local attendable range
        can_attend_local = (kv_window_idx >= local_min_kv_window) & (kv_window_idx <= local_max_kv_window)

        # Condition 2: Sink window attention
        # If sink_window is True, all queries can also attend to the very first key/value window (index 0)
        if sink_window:
            can_attend_sink = kv_window_idx == 0
            # Combine local attention with sink attention using logical OR
            final_mask = can_attend_local | can_attend_sink
        else:
            final_mask = can_attend_local

        if dropped_frame_pairs_tensor is not None:
            dropped_kv = dropped_frame_pairs_tensor[q_window_idx, kv_window_idx]
            same_frame = kv_window_idx == q_window_idx
            final_mask = final_mask & (~dropped_kv | same_frame)

        return final_mask

    return mask_mod


# mask_mod = custom_mask_mod_with_params(
#     block_size=window_size,
#     look_forward=0,
#     look_backward=8,
#     sink_window=False,
# )
# block_mask = create_block_mask_cached(
#             mask_mod=mask_mod,
#             B=1,
#             H=1,
#             M=seq_len,
#             N=seq_len,
#             device=x.device,
#         )
