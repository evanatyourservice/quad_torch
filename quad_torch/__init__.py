import torch

from quad_torch._psgd import PSGD


class Procrustes(PSGD):
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.001,
        lr_style: str | None = "adam",
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        max_size_dense: int = 16384,
        max_skew_dense: float = 1.0,
        preconditioner_lr: float = 0.7,
        preconditioner_init_scale: float | None = None,
        noise_scale: float = 1e-9,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            params=params,
            lr=lr,
            lr_style=lr_style,
            momentum=momentum,
            weight_decay=weight_decay,
            psgd_type="procrustes",
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            noise_scale=noise_scale,
            dtype=dtype,
        )


class QUAD(PSGD):
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.001,
        lr_style: str | None = "adam",
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        max_size_dense: int = 16384,
        max_skew_dense: float = 1.0,
        preconditioner_lr: float = 0.7,
        preconditioner_init_scale: float | None = None,
        noise_scale: float = 1e-9,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            params=params,
            lr=lr,
            lr_style=lr_style,
            momentum=momentum,
            weight_decay=weight_decay,
            psgd_type="quad",
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            noise_scale=noise_scale,
            dtype=dtype,
        )


__all__ = ["Procrustes", "QUAD"]

Procrustes.__doc__ = PSGD.__doc__
QUAD.__doc__ = PSGD.__doc__
