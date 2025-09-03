import math
import torch


class QUAD(torch.optim.Optimizer):
    """PSGD-QUAD optimizer.

    This optimizer is best used without gradient clipping.

    Args:
        params: list of parameters to optimize
        lr: learning rate
        lr_style: "adam" (default) or None. "adam" scales update to match adam's behavior, None uses
            original PSGD scaling (RMS=1.0).
        momentum: momentum beta
        weight_decay: weight decay
        max_size_dense: dimensions larger than this will have diagonal preconditioners, otherwise
            dense.
        max_skew_dense: dimensions with skew larger than this compared to the other dimension will
            have diagonal preconditioners, otherwise dense.
        preconditioner_lr: preconditioner learning rate
        noise_scale: scale of noise added to gradients.
        dtype: dtype for all computations and states in QUAD. None defaults to dtype of gradients.
    """
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.001,
        lr_style: str | None = "adam",
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        max_size_dense: int = 8192,
        max_skew_dense: float = 1.0,
        preconditioner_lr: float = 0.7,
        noise_scale: float = 1e-9,
        dtype: torch.dtype | None = None,
    ):
        defaults = dict(
            lr=lr,
            lr_style=lr_style,
            momentum=momentum,
            weight_decay=weight_decay,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            preconditioner_lr=preconditioner_lr,
            noise_scale=noise_scale,
            dtype=dtype,
        )
        super().__init__(params, defaults)        

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        momentum_buffers,
        merged_shapes,
        Qs,
        Ls,
        diags,
        state_steps,
    ):
        group_dtype = group['dtype']
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad if group_dtype is None else p.grad.to(dtype=group_dtype))

        for p, g in zip(params_with_grad, grads):
            state = self.state[p]
            dtype = g.dtype

            if "momentum_buffer" not in state:
                state["step"] = torch.tensor(0, dtype=torch.int32, device=g.device)
                state["momentum_buffer"] = g.clone()
                state["merged_shape"] = merge_dims(state["momentum_buffer"])
                g_reshaped = state["momentum_buffer"].view(state["merged_shape"])
                scale = (torch.mean(torch.abs(g_reshaped)) + group["noise_scale"])**(-1/(4 if len(g_reshaped.shape) > 1 else 2))
                if g_reshaped.ndim <= 1:
                    state["Q"] = [scale * torch.ones_like(g_reshaped, dtype=dtype)]
                    state["L"] = [torch.zeros([], dtype=dtype, device=g_reshaped.device)]
                    state["diag"] = [True]
                else:
                    Qs_new = []
                    Ls_new = []
                    diag_new = []
                    for size in g_reshaped.shape:
                        if size > group["max_size_dense"] or size**2 > group["max_skew_dense"] * g_reshaped.numel():
                            Qs_new.append(scale * torch.ones(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=dtype, device=g_reshaped.device))
                            diag_new.append(True)
                        else:
                            Qs_new.append(scale * torch.eye(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=dtype, device=g_reshaped.device))
                            diag_new.append(False)
                    state["Q"] = Qs_new
                    state["L"] = Ls_new
                    state["diag"] = diag_new
    
            momentum_buffers.append(state['momentum_buffer'])
            merged_shapes.append(state["merged_shape"])
            Qs.append(state["Q"])
            Ls.append(state["L"])
            diags.append(state["diag"])
            state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            momentum_buffers: list[torch.Tensor] = []
            merged_shapes: list[tuple] = []
            Qs: list[list | None] = []
            Ls: list[list | None] = []
            diags: list[list | None] = []
            state_steps: list[int] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                momentum_buffers,
                merged_shapes,
                Qs,
                Ls,
                diags,
                state_steps,
            )

            if len(params_with_grad) == 0:
                continue

            torch._foreach_lerp_(momentum_buffers, grads, 1 - group['momentum'])

            preconditioned_grads = []
            for (p, g, merged_shape, Q, L, diag) in zip(
                params_with_grad, momentum_buffers, merged_shapes, Qs, Ls, diags
            ):
                state = self.state[p]
                state["step"] += 1
                original_shape = g.shape
                g_reshaped = g.view(merged_shape)

                if g_reshaped.ndim <= 1:
                    g_preconditioned = update_diag_solo(
                        Q[0], L[0], g_reshaped, group["preconditioner_lr"], state["step"], group["noise_scale"]
                    )
                else:
                    if state["step"] % 50 == 0:
                        balance_preconditioners(Q)
                    
                    if not diag[0] and not diag[1]:
                        g_preconditioned = precondition_DD(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            noise_scale=group["noise_scale"]
                        )
                    elif diag[0] and not diag[1]:
                        g_preconditioned = precondition_dD(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            noise_scale=group["noise_scale"]
                        )
                    elif not diag[0] and diag[1]:
                        g_preconditioned = precondition_Dd(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            noise_scale=group["noise_scale"]
                        )
                    else:
                        g_preconditioned = precondition_dd(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            noise_scale=group["noise_scale"]
                        )

                clip_update(g_preconditioned)            
                if g_preconditioned.shape != original_shape:
                    g_preconditioned = g_preconditioned.view(original_shape)   
                preconditioned_grads.append(g_preconditioned.to(dtype=p.dtype))

            if group["weight_decay"] > 0:
                torch._foreach_mul_(params_with_grad, 1 - group["lr"] * group["weight_decay"])
            
            torch._foreach_add_(
                params_with_grad,
                preconditioned_grads,
                # adam lr can be simulated by scaling down psgd update
                alpha=-group["lr"] / 5.0 if group["lr_style"] == "adam" else -group["lr"]
            )
        return loss


@torch.compile(fullgraph=True)
def balance_preconditioners(Qs):
    ql, qr = Qs
    max_l = torch.amax(torch.abs(ql))
    max_r = torch.amax(torch.abs(qr))
    gmean = torch.sqrt(max_l * max_r)
    ql.mul_(gmean / max_l)
    qr.mul_(gmean / max_r)


def get_precond_lr(lr, step):
    return torch.clamp(lr * torch.rsqrt(1.0 + step / 10000.0), min=0.3)


def add_noise(x, scale):
    return x + torch.randn_like(x) * scale


@torch.compile(fullgraph=True)
def update_diag_solo(Q, L, G, precond_lr, step, noise_scale):
    Pg = Q * Q * add_noise(G, scale=noise_scale)
    term1 = Pg * Pg
    term2 = 1.0
    ell = torch.amax(term1) + term2
    L.copy_(torch.max(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = get_precond_lr(precond_lr, step) / (2 * L)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)
    return Q * Q * G


def _diag_update(term1, term2, L, Q, precond_lr, step):
    ell = torch.amax(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = get_precond_lr(precond_lr, step) / (2 * L)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)


def lb(A_outer: torch.Tensor):
    max_abs = A_outer.diagonal().max()
    A = A_outer / max_abs
    j = torch.argmax(torch.sum(A * A, dim=1))
    x = A.mv(A.index_select(0, j))
    return A.mv(x / x.norm()).norm() * max_abs


def _dense_update(term1, term2, L, Q, precond_lr, step):
    ell = lb(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = get_precond_lr(precond_lr, step) / (2 * L)
    p = Q - lr_over_2L * (term1 @ Q - term2 * Q)
    p = p - lr_over_2L * (p @ term1 - p * term2)
    Q.copy_((p + p.T) / 2)


@torch.compile(fullgraph=True)
def precondition_dd(Ql, Qr, Ll, Lr, G, precond_lr, step, noise_scale):
    """Diagonal-diagonal preconditioning."""
    Pg = (Ql * Ql).unsqueeze(1) * add_noise(G, scale=noise_scale) * (Qr * Qr).unsqueeze(0)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = G.numel() / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = G.numel() / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return (Ql * Ql).unsqueeze(1) * G * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_dD(Ql, Qr, Ll, Lr, G, precond_lr, step, noise_scale):
    """Diagonal-dense preconditioning."""
    noiseG = add_noise(G, scale=noise_scale)
    Pg = (Ql * Ql).unsqueeze(1) * torch.linalg.multi_dot([noiseG, Qr, Qr])
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = G.numel() / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = G.numel() / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return (Ql * Ql).unsqueeze(1) * torch.linalg.multi_dot([G, Qr, Qr])


@torch.compile(fullgraph=True)
def precondition_Dd(Ql, Qr, Ll, Lr, G, precond_lr, step, noise_scale):
    """Dense-diagonal preconditioning."""
    noiseG = add_noise(G, scale=noise_scale)
    Pg = torch.linalg.multi_dot([Ql, Ql, noiseG]) * (Qr * Qr).unsqueeze(0)
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = G.numel() / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = G.numel() / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return torch.linalg.multi_dot([Ql, Ql, G]) * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_DD(Ql, Qr, Ll, Lr, G, precond_lr, step, noise_scale):
    """Dense-dense preconditioning."""
    noiseG = add_noise(G, scale=noise_scale)
    Pg = torch.linalg.multi_dot([Ql, Ql, noiseG, Qr, Qr])
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = G.numel() / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = G.numel() / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return torch.linalg.multi_dot([Ql, Ql, G, Qr, Qr])


@torch.compile(fullgraph=True)
def clip_update(G):
    G.mul_(1.05 / G.square().mean().sqrt().clamp(min=1.05))             


def merge_dims(tensor: torch.Tensor):
    """Merge tensor dimensions into the most square matrix."""
    if tensor.ndim < 2:
        return tensor.shape
    if math.prod(tensor.shape) == math.max(tensor.shape):
        return (math.max(tensor.shape),)
    if len(tensor.shape) == 2:
        return tensor.shape
    dims = list(tensor.shape)
    best_ratio = float('inf')
    best_split = 1
    for split_idx in range(1, len(dims)):
        left_prod = math.prod(dims[:split_idx])
        right_prod = math.prod(dims[split_idx:])
        ratio = max(left_prod, right_prod) / min(left_prod, right_prod)
        if ratio < best_ratio:
            best_ratio = ratio
            best_split = split_idx
    return math.prod(dims[:best_split]), math.prod(dims[best_split:])
