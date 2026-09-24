"""Entry-point module for `cuda_frac`, separate so it can be blocked on its own (SPEC 1.1)."""

from .plugin import CUDAFracPlugin

__all__ = ("CUDAFracPlugin",)
