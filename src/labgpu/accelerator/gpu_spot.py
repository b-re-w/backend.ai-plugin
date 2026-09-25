"""Entry-point module for the per-model spot plugins `gpu_spot_N` (SPEC 2.12)."""

from .spot_plugin import SpotSlotPlugin1, SpotSlotPlugin2, SpotSlotPlugin3, SpotSlotPlugin4

__all__ = ("SpotSlotPlugin1", "SpotSlotPlugin2", "SpotSlotPlugin3", "SpotSlotPlugin4")
