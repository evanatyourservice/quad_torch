import math
import torch


class QUAD(torch.optim.Optimizer):
    """PSGD-QUAD optimizer.

    Args:
        params: list of parameters to optimize
        lr: learning rate
        lr_style: "adam" (default), "mu-p", or None, "adam" scales update norm to match adam's,
            "mu-p" scales update norm according to sqrt(grad.shape[-2]), None uses PSGD scaling of 
            RMS=1.0.
        momentum: momentum beta
        weight_decay: weight decay
        preconditioner_lr: preconditioner learning rate
        max_size_dense: dimensions larger than this will have diagonal preconditioners, otherwise
            dense.
        max_skew_dense: dimensions with skew larger than this compared to the other dimension will
            have diagonal preconditioners, otherwise dense.
        noise_scale: scale of noise added to gradients.
        normalize_grads: normalize incoming gradients to unit norm.
        dtype: dtype for all computations and states in QUAD. None defaults to dtype of gradients.
    """
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.001,
        lr_style: str | None = "adam",
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        preconditioner_lr: float = 0.6,
        max_size_dense: int = 8192,
        max_skew_dense: float = 1.0,
        noise_scale: float = 1e-8,
        normalize_grads: bool = False,
        dtype: torch.dtype | None = None,
    ):
        defaults = dict(
            lr=lr,
            lr_style=lr_style,
            momentum=momentum,
            weight_decay=weight_decay,
            preconditioner_lr=preconditioner_lr,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            noise_scale=noise_scale,
            normalize_grads=normalize_grads,
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
        mu_ps,
        state_steps,
    ):
        group_dtype = group['dtype']
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad if group_dtype is None else p.grad.to(dtype=group_dtype))
    
        if group["normalize_grads"]:
            torch._foreach_div_(grads, torch._foreach_add_(torch._foreach_norm(grads), 1e-6))
    
        for p, g in zip(params_with_grad, grads):
            state = self.state[p]
            dtype = g.dtype

            # mu-p is calculated from G[-2] dim, but this could be based on any dim or size
            mu_ps.append(g.shape[-2] if len(g.shape) > 1 else None)
    
            if "momentum_buffer" not in state:
                state["step"] = torch.tensor(0, dtype=torch.int32, device=g.device)
                state["momentum_buffer"] = g.clone()
                state["merged_shape"] = merge_dims(state["momentum_buffer"])
                g_reshaped = state["momentum_buffer"].view(state["merged_shape"])
                scale = (torch.mean(torch.abs(g_reshaped)) + group["noise_scale"])**(-1/(4 if len(g_reshaped.shape) > 1 else 2))
                if g_reshaped.ndim <= 1:
                    state["Q"] = [scale * torch.ones_like(g_reshaped, dtype=dtype)]
                    state["L"] = [torch.zeros([], dtype=torch.float32, device=g_reshaped.device)]
                    state["diag"] = [True]
                else:
                    Qs_new = []
                    Ls_new = []
                    diag_new = []
                    for size in g_reshaped.shape:
                        if size > group["max_size_dense"] or size**2 > group["max_skew_dense"] * g_reshaped.numel():
                            Qs_new.append(scale * torch.ones(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=torch.float32, device=g_reshaped.device))
                            diag_new.append(True)
                        else:
                            Qs_new.append(scale * torch.eye(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=torch.float32, device=g_reshaped.device))
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
            mu_ps: list[float] = []
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
                mu_ps,
                state_steps,
            )

            if len(params_with_grad) == 0:
                continue

            torch._foreach_lerp_(momentum_buffers, grads, 1 - group['momentum'])

            preconditioned_grads = []
            for (p, g, merged_shape, Q, L, diag, mu_p_size) in zip(
                params_with_grad, momentum_buffers, merged_shapes,
                Qs, Ls, diags, mu_ps
            ):
                dtype = g.dtype
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
                        ql, qr = Q
                        max_l = torch.amax(torch.abs(ql))
                        max_r = torch.amax(torch.abs(qr))
                        gmean = torch.sqrt(max_l * max_r)
                        ql.mul_(gmean / max_l)
                        qr.mul_(gmean / max_r)
                    
                    # we can do mu-p by simply changing the target scale of the preconditioner
                    # (see above in self._init_group)
                    term2_target = mu_p_size if group["lr_style"] == "mu-p" else g_reshaped.numel()

                    if not diag[0] and not diag[1]:
                        g_preconditioned = precondition_DD(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target,
                            noise_scale=group["noise_scale"]
                        )
                    elif diag[0] and not diag[1]:
                        g_preconditioned = precondition_dD(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target,
                            noise_scale=group["noise_scale"]
                        )
                    elif not diag[0] and diag[1]:
                        g_preconditioned = precondition_Dd(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target,
                            noise_scale=group["noise_scale"]
                        )
                    else:
                        g_preconditioned = precondition_dd(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target,
                            noise_scale=group["noise_scale"]
                        )

                original_shape = p.grad.shape
                if original_shape != g_preconditioned.shape:
                    g_preconditioned = g_preconditioned.view(original_shape)
                
                assert g_preconditioned.dtype == dtype, f"{g_preconditioned.dtype} != {dtype} for param shape {original_shape}"
                
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


def get_precond_lr(lr, step):
    return torch.clamp(lr * torch.rsqrt(1.0 + step / 10000.0), min=0.1)


def add_noise(x, scale=1e-8):
    return x + torch.randn_like(x) * scale


@torch.compile(fullgraph=True)
def update_diag_solo(Q, L, G, precond_lr, step, noise_scale: float = 1e-8):
    Pg = Q * Q * add_noise(G, scale=noise_scale)
    term1 = Pg * Pg
    term2 = 1.0
    ell = (torch.amax(term1) + term2).to(torch.float32)
    L.copy_(torch.max(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = (get_precond_lr(precond_lr, step) / (2 * L)).to(Q.dtype)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)
    return Q * Q * G


def _diag_update(term1, term2, L, Q, precond_lr, step):
    ell = (torch.amax(term1) + term2).to(torch.float32)
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = (get_precond_lr(precond_lr, step) / (2 * L)).to(Q.dtype)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)


def lb(A_outer: torch.Tensor):
    max_abs = A_outer.diagonal().max()
    A = A_outer / max_abs
    j = torch.argmax(torch.sum(A * A, dim=1))
    x = A.index_select(0, j).view(-1)
    x = A.mv(x)
    x = x / x.norm()
    return A.mv(x).norm() * max_abs


def _dense_update(term1, term2, L, Q, precond_lr, step):
    ell = (lb(term1) + term2).to(torch.float32)
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = (get_precond_lr(precond_lr, step) / (2 * L)).to(Q.dtype)
    p = Q - lr_over_2L * (term1 @ Q - term2 * Q)
    p = p - lr_over_2L * (p @ term1 - p * term2)
    Q.copy_((p + p.T) / 2)


@torch.compile(fullgraph=True)
def precondition_dd(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target, noise_scale: float = 1e-8):
    """Diagonal-diagonal preconditioning."""
    Pg = (Ql * Ql).unsqueeze(1) * add_noise(G, scale=noise_scale) * (Qr * Qr).unsqueeze(0)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = term2_target / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = term2_target / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return (Ql * Ql).unsqueeze(1) * G * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_dD(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target, noise_scale: float = 1e-8):
    """Diagonal-dense preconditioning."""
    Pg = (Ql * Ql).unsqueeze(1) * add_noise(G, scale=noise_scale) @ (Qr.T @ Qr)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = term2_target / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = term2_target / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return (Ql * Ql).unsqueeze(1) * G @ (Qr.T @ Qr)


@torch.compile(fullgraph=True)
def precondition_Dd(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target, noise_scale: float = 1e-8):
    """Dense-diagonal preconditioning."""
    Pg = (Ql.T @ Ql) @ add_noise(G, scale=noise_scale) * (Qr * Qr).unsqueeze(0)
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = term2_target / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = term2_target / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return (Ql.T @ Ql) @ G * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_DD(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target, noise_scale: float = 1e-8):
    """Dense-dense preconditioning."""
    Pg = (Ql.T @ Ql) @ add_noise(G, scale=noise_scale) @ (Qr.T @ Qr)
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = term2_target / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = term2_target / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return (Ql.T @ Ql) @ G @ (Qr.T @ Qr)


def merge_dims(tensor):
    """Merge tensor dimensions into the most square matrix."""
    if tensor.ndim <= 2:
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
