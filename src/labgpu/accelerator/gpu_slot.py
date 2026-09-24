"""Entry-point module for the per-model `gpu_slot_N` plugins (SPEC 1.1, 1.11)."""

from .plugin import GpuSlotPlugin1, GpuSlotPlugin2, GpuSlotPlugin3, GpuSlotPlugin4

__all__ = ("GpuSlotPlugin1", "GpuSlotPlugin2", "GpuSlotPlugin3", "GpuSlotPlugin4")
