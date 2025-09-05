import math
import torch


class QUAD(torch.optim.Optimizer):
    """PSGD-QUAD optimizer.

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
            state = self.state[p]
            if group_dtype is not None and p.grad.dtype != group_dtype:
                g = p.grad.to(dtype=group_dtype)
            else:
                g = p.grad
            grads.append(g)
            dtype = g.dtype
            if "momentum_buffer" not in state:
                state["step"] = torch.tensor(0, dtype=torch.int32, device=g.device)
                state["momentum_buffer"] = g.clone()
                state["merged_shape"] = merge_dims(state["momentum_buffer"])
                g_reshaped = state["momentum_buffer"].view(state["merged_shape"])
                scale = (torch.mean(g_reshaped**2) + group["noise_scale"]**2)**(-1/(8 if len(g_reshaped.shape) > 1 else 4))
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
                lr_precond = get_precond_lr(group["preconditioner_lr"], state["step"])
                if g_reshaped.ndim <= 1:
                    g_preconditioned = update_diag_solo(
                        Q[0], L[0], g_reshaped, lr_precond, group["noise_scale"]
                    )
                else:
                    if state["step"] % 100 == 0:
                        balance_preconditioners(Q)
                    if not any(diag):
                        g_preconditioned = precondition_DD(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"]
                        )
                    elif diag[0] and not diag[1]:
                        g_preconditioned = precondition_dD(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"]
                        )
                    elif not diag[0] and diag[1]:
                        g_preconditioned = precondition_Dd(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"]
                        )
                    else:
                        g_preconditioned = precondition_dd(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"]
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
                # adam's update RMS can be simulated by scaling down psgd's update RMS
                # 1.0 (quad) / 5.0 = 0.2 (≈adam)
                alpha=-group["lr"] / 5.0 if group["lr_style"] == "adam" else -group["lr"]
            )
        return loss


def get_precond_lr(lr, step):
    # Decaying from some higher number down to 0.3 improves performance a bit vs. static 
    # preconditioner LR. 0.3 minimum seems to be fair sweet spot, allowing tighter convergence 
    # (nice to have) without loss of isotropy (most important).
    return torch.clamp(lr * torch.rsqrt(1.0 + step / 10000.0), min=0.3)


def add_noise(x, scale):
    return x + torch.randn_like(x) * scale


@torch.compile(fullgraph=True)
def balance_preconditioners(Qs):
    ql, qr = Qs
    max_l = torch.amax(torch.abs(ql))
    max_r = torch.amax(torch.abs(qr))
    gmean = torch.sqrt(max_l * max_r)
    ql.mul_(gmean / max_l)
    qr.mul_(gmean / max_r)


@torch.compile(fullgraph=True)
def update_diag_solo(Q, L, G, lr_precond, noise_scale):
    Pg = Q * Q * add_noise(G, scale=noise_scale)
    term1 = Pg * Pg
    term2 = 1.0
    ell = torch.amax(term1) + term2
    L.copy_(torch.max(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = lr_precond / (2 * L)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)
    return Q * Q * G


def _diag_update(term1, term2, L, Q, lr_precond):
    ell = torch.amax(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = lr_precond / (2 * L)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)


def lb(A_outer: torch.Tensor):
    max_abs = A_outer.diagonal().max()
    A = A_outer / max_abs
    j = torch.argmax(torch.sum(A * A, dim=1))
    x = A.mv(A.index_select(0, j))
    return A.mv(x / x.norm()).norm() * max_abs


def _dense_update(term1, term2, L, Q, lr_precond):
    ell = lb(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = lr_precond / (2 * L)
    p = Q - lr_over_2L * (term1 @ Q - term2 * Q)
    p = p - lr_over_2L * (p @ term1 - p * term2)
    Q.copy_((p + p.T) / 2)


@torch.compile(fullgraph=True)
def precondition_dd(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale):
    """Diagonal-diagonal preconditioning."""
    Pg = (Ql * Ql).unsqueeze(1) * add_noise(G, scale=noise_scale) * (Qr * Qr).unsqueeze(0)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = G.numel() / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = G.numel() / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return (Ql * Ql).unsqueeze(1) * G * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_dD(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale):
    """Diagonal-dense preconditioning."""
    noiseG = add_noise(G, scale=noise_scale)
    Pg = (Ql * Ql).unsqueeze(1) * torch.linalg.multi_dot([noiseG, Qr, Qr])
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = G.numel() / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = G.numel() / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return (Ql * Ql).unsqueeze(1) * torch.linalg.multi_dot([G, Qr, Qr])


@torch.compile(fullgraph=True)
def precondition_Dd(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale):
    """Dense-diagonal preconditioning."""
    noiseG = add_noise(G, scale=noise_scale)
    Pg = torch.linalg.multi_dot([Ql, Ql, noiseG]) * (Qr * Qr).unsqueeze(0)
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = G.numel() / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = G.numel() / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return torch.linalg.multi_dot([Ql, Ql, G]) * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_DD(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale):
    """Dense-dense preconditioning."""
    noiseG = add_noise(G, scale=noise_scale)
    Pg = torch.linalg.multi_dot([Ql, Ql, noiseG, Qr, Qr])
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = G.numel() / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = G.numel() / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return torch.linalg.multi_dot([Ql, Ql, G, Qr, Qr])


@torch.compile(fullgraph=True)
def clip_update(g):
    # QUAD is best used without incoming gradient clipping or normalization so that the 
    # preconditioner can see scale differences in the gradient between train steps. We clip the 
    # final update to guard against surprisingly large gradients in case a single preconditioner 
    # update is not enough to fully normalize the gradient. PSGD update should be around 1.0 RMS 
    # so let's clip at 1.05.
    g.mul_(1.05 / g.square().mean().sqrt().clamp(min=1.05))             


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
