"""R3 wrapper internals.

These mixins are implementation details of :class:`R3.models.r3.R3`, not
standalone model utilities.
"""

from R3.models.r3_wrapper.online_inference import R3OnlineInferenceMixin
from R3.models.r3_wrapper.outputs import R3OutputMixin
from R3.models.r3_wrapper.setup import R3SetupMixin

__all__ = [
    "R3OnlineInferenceMixin",
    "R3OutputMixin",
    "R3SetupMixin",
]
