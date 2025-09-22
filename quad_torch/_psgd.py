import math
import torch


class PSGD(torch.optim.Optimizer):
    """PSGD optimizer.

    Args:
        params: list of parameters to optimize.
        lr: learning rate.
        lr_style: "adam" (default) or None. "adam" scales update to match adam's behavior, None uses
            original PSGD scaling (RMS=1.0).
        momentum: momentum beta.
        weight_decay: weight decay.
        psgd_type: "quad" or "procrustes". Procrustes is a more exact implementation that instead updates 
            Q using Q0.5EQ1.5.
        max_size_dense: dimensions larger than this will have diagonal preconditioners, otherwise
            dense.
        max_skew_dense: dimensions with skew larger than this compared to the other dimension will
            have diagonal preconditioners, otherwise dense.
        preconditioner_lr: preconditioner learning rate.
        preconditioner_init_scale: scale of initial preconditioner values.
        noise_scale: scale of noise added to gradients.
        dtype: dtype for all computations and states in QUAD. None defaults to dtype of gradients layer-wise.
    """
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.001,
        lr_style: str | None = "adam",
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        psgd_type: str = "procrustes",  # "quad" or "procrustes"
        max_size_dense: int = 16384,
        max_skew_dense: float = 1.0,
        preconditioner_lr: float = 0.7,
        preconditioner_init_scale: float | None = None,
        noise_scale: float = 1e-9,
        dtype: torch.dtype | None = None,
    ):
        self.solo_diag_fn = update_solo_diag_quad if psgd_type == "quad" else update_solo_diag_procrustes
        self.diag_fn = _update_diag_quad if psgd_type == "quad" else _update_diag_procrustes
        self.dense_fn = _update_dense_quad if psgd_type == "quad" else _update_dense_procrustes

        defaults = dict(
            lr=lr,
            lr_style=lr_style,
            momentum=momentum,
            weight_decay=weight_decay,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            noise_scale=noise_scale,
            dtype=dtype,
            do_init=True,
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
            if group_dtype is not None and p.grad.dtype != group_dtype:
                g = p.grad.to(dtype=group_dtype)
            else:
                g = p.grad
            grads.append(g)

        if group["do_init"]:
            init_scale = group["preconditioner_init_scale"]
            if init_scale is None:
                second_moment_max = max([torch.mean(torch.abs(g)**2) for g in grads])
                init_scale = (second_moment_max + (group["noise_scale"]**2)) ** (-1/4)

        for p, g in zip(params_with_grad, grads):
            state = self.state[p]
            dtype = g.dtype
            if group["do_init"]:
                state["step"] = torch.tensor(0, dtype=torch.int32, device=g.device)
                state["momentum_buffer"] = g.clone()
                state["merged_shape"] = _merge_dims(state["momentum_buffer"])
                g_reshaped = state["momentum_buffer"].view(state["merged_shape"])
                if g_reshaped.ndim <= 1:
                    state["Q"] = [init_scale * torch.ones_like(g_reshaped, dtype=dtype)]
                    state["L"] = [torch.zeros([], dtype=dtype, device=g_reshaped.device)]
                    state["diag"] = [True]
                else:
                    Qs_new = []
                    Ls_new = []
                    diag_new = []
                    for size in g_reshaped.shape:
                        if size > group["max_size_dense"] or size**2 > group["max_skew_dense"] * g_reshaped.numel():
                            Qs_new.append(init_scale * torch.ones(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=dtype, device=g_reshaped.device))
                            diag_new.append(True)
                        else:
                            Qs_new.append(init_scale * torch.eye(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=dtype, device=g_reshaped.device))
                            diag_new.append(False)
                    state["Q"] = Qs_new
                    state["L"] = Ls_new
                    state["diag"] = diag_new
                _print_preconditioner_summary(
                    original_shape=g.shape, merged_shape=state["merged_shape"], diagonal_flags=state["diag"]
                )
            momentum_buffers.append(state['momentum_buffer'])
            merged_shapes.append(state["merged_shape"])
            Qs.append(state["Q"])
            Ls.append(state["L"])
            diags.append(state["diag"])
            state_steps.append(state["step"])
        group["do_init"] = False

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
                    g_preconditioned = self.solo_diag_fn(
                        Q[0], L[0], g_reshaped, lr_precond, group["noise_scale"]
                    )
                else:
                    if state["step"] % 100 == 0:
                        _balance_preconditioners(Q)
                    if not any(diag):
                        g_preconditioned = precondition_DD(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"],
                            dense_fn=self.dense_fn
                        )
                    elif diag[0] and not diag[1]:
                        g_preconditioned = precondition_dD(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"],
                            diag_fn=self.diag_fn, dense_fn=self.dense_fn
                        )
                    elif not diag[0] and diag[1]:
                        g_preconditioned = precondition_Dd(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"],
                            diag_fn=self.diag_fn, dense_fn=self.dense_fn
                        )
                    else:
                        g_preconditioned = precondition_dd(
                            *Q, *L, G=g_reshaped, lr_precond=lr_precond, noise_scale=group["noise_scale"],
                            diag_fn=self.diag_fn
                        )
                _clip_update(g_preconditioned)            
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
    # Decaying from some higher number down to min_lr improves performance a bit vs. static 
    # preconditioner LR. min_lr minimum seems to be fair sweet spot, allowing tighter convergence 
    # (nice to have) without loss of isotropy (most important).
    min_lr = 0.3
    return torch.clamp(lr * torch.rsqrt(1.0 + step / 10000.0), min=min_lr)


def _add_noise(x, scale):
    return x + torch.randn_like(x) * scale


@torch.compile(fullgraph=True)
def _balance_preconditioners(Qs):
    ql, qr = Qs
    max_l = torch.amax(torch.abs(ql))
    max_r = torch.amax(torch.abs(qr))
    gmean = torch.sqrt(max_l * max_r)
    ql.mul_(gmean / max_l)
    qr.mul_(gmean / max_r)


@torch.compile(fullgraph=True)
def update_solo_diag_quad(Q, L, G, lr_precond, noise_scale):
    Pg = Q * Q * _add_noise(G, scale=noise_scale)
    term1 = Pg * Pg
    term2 = 1.0
    ell = torch.amax(term1) + term2
    L.copy_(torch.max(0.95 * L + 0.05 * ell, ell))
    gain = 1 - lr_precond / (2 * L) * (term1 - term2)
    Q.mul_(gain * gain)
    return Q * Q * G


@torch.compile(fullgraph=True)
def update_solo_diag_procrustes(Q, L, G, lr_precond, noise_scale):
    Pg = Q * Q * _add_noise(G, scale=noise_scale)
    term1 = Pg * Pg
    term2 = 1.0
    ell = torch.amax(term1) + term2
    L.copy_(torch.max(0.95 * L + 0.05 * ell, ell))
    Q.mul_(1 - lr_precond / L * (term1 - term2))
    return Q * Q * G


def _update_diag_quad(term1, term2, L, Q, lr_precond):
    ell = torch.amax(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    gain = 1 - lr_precond / (2 * L) * (term1 - term2)
    Q.mul_(gain * gain)


def _update_diag_procrustes(term1, term2, L, Q, lr_precond):
    ell = torch.amax(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    Q.mul_(1 - lr_precond / L * (term1 - term2))


def _norm_lower_bound(A, k=4, iters=5, skh=False):
    if skh:
        max_abs = A.abs().amax()
    else:
        max_abs = A.diagonal().amax()
    A = A / max_abs
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    a = torch.index_select(A, 0, j).squeeze()
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    V = a + torch.sign(torch.sum(a * V, dim=1, keepdim=True)) * V
    for _ in range(iters):
        V = V / V.abs().amax()
        V = V @ A   
    V = V / torch.linalg.vector_norm(V, dim=1, keepdim=True) + 1e-9
    return torch.amax(torch.linalg.vector_norm(V @ A, dim=1)) * max_abs


def _update_dense_quad(term1, term2, L, Q, lr_precond):
    ell = _norm_lower_bound(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    lr_over_2L = lr_precond / (2 * L)
    p = Q - lr_over_2L * (term1 @ Q - term2 * Q)
    p = p - lr_over_2L * (p @ term1 - p * term2)
    Q.copy_((p + p.T) / 2)


def _procrustes_step(Q, max_step_size=1/8):
    R = Q.T - Q
    max_abs = R.abs().amax()

    def inner(R):
        R = R / max_abs
        RQ = R @ Q
        tr_RQ = RQ.diagonal().sum()
        
        def do_rotation():
            # rotate Q as exp(a R) Q ~ (I + a R + a^2 R^2/2) Q with an optimal a
            a = max_step_size / _norm_lower_bound(R, skh=True)
            RRQ = R @ RQ
            tr_RRQ = RRQ.diagonal().sum()
            a = torch.where(tr_RRQ < 0, min(a, -tr_RQ / tr_RRQ), a)
            return a * (RQ + 0.5 * a * RRQ)
        
        return torch.where(tr_RQ > 0, do_rotation(), torch.zeros_like(Q))

    Q.add_(torch.where(max_abs > torch.finfo(max_abs.dtype).smallest_normal, inner(R), torch.zeros_like(Q)))


def _update_dense_procrustes(term1, term2, L, Q, lr_precond):
    ell = _norm_lower_bound(term1) + term2
    L.copy_(torch.maximum(0.95 * L + 0.05 * ell, ell))
    Q.sub_(lr_precond / L * (term1 @ Q - term2 * Q))
    _procrustes_step(Q)


@torch.compile(fullgraph=True)
def precondition_dd(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale, diag_fn):
    """Diagonal-diagonal preconditioning."""
    Pg = (Ql * Ql).unsqueeze(1) * _add_noise(G, scale=noise_scale) * (Qr * Qr).unsqueeze(0)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = G.numel() / Ql.shape[0]
    diag_fn(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = G.numel() / Qr.shape[0]
    diag_fn(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return (Ql * Ql).unsqueeze(1) * G * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_dD(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale, diag_fn, dense_fn):
    """Diagonal-dense preconditioning."""
    noiseG = _add_noise(G, scale=noise_scale)
    Pg = (Ql * Ql).unsqueeze(1) * torch.linalg.multi_dot([noiseG, Qr, Qr])
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = G.numel() / Ql.shape[0]
    diag_fn(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = G.numel() / Qr.shape[0]
    dense_fn(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return (Ql * Ql).unsqueeze(1) * torch.linalg.multi_dot([G, Qr, Qr])


@torch.compile(fullgraph=True)
def precondition_Dd(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale, diag_fn, dense_fn):
    """Dense-diagonal preconditioning."""
    noiseG = _add_noise(G, scale=noise_scale)
    Pg = torch.linalg.multi_dot([Ql, Ql, noiseG]) * (Qr * Qr).unsqueeze(0)
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = G.numel() / Ql.shape[0]
    dense_fn(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = G.numel() / Qr.shape[0]
    diag_fn(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return torch.linalg.multi_dot([Ql, Ql, G]) * (Qr * Qr).unsqueeze(0)


@torch.compile(fullgraph=True)
def precondition_DD(Ql, Qr, Ll, Lr, G, lr_precond, noise_scale, dense_fn):
    """Dense-dense preconditioning."""
    noiseG = _add_noise(G, scale=noise_scale)
    Pg = torch.linalg.multi_dot([Ql, Ql, noiseG, Qr, Qr])
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = G.numel() / Ql.shape[0]
    dense_fn(term1_l, term2_l, Ll, Ql, lr_precond)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = G.numel() / Qr.shape[0]
    dense_fn(term1_r, term2_r, Lr, Qr, lr_precond)
    
    return torch.linalg.multi_dot([Ql, Ql, G, Qr, Qr])


@torch.compile(fullgraph=True)
def _clip_update(g):
    # QUAD is best used without incoming gradient clipping or normalization so that the 
    # preconditioner can see scale differences in the gradient between train steps. We clip the 
    # final update to guard against surprisingly large gradients in case a single preconditioner 
    # update is not enough to fully normalize the gradient. PSGD update should be around 1.0 RMS 
    # so let's clip at 1.1.
    g.mul_(1.1 / g.square().mean().sqrt().clamp(min=1.1))   


def _merge_dims(tensor):
    if tensor.ndim < 2:
        return tensor.shape
    if math.prod(tensor.shape) == max(tensor.shape):
        return (max(tensor.shape),)
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


def _print_preconditioner_summary(original_shape, merged_shape, diagonal_flags):
    original_shape_tuple = tuple(original_shape)
    merged_shape_tuple = tuple(merged_shape)
    preconditioner_types = ["diagonal" if is_diag else "dense" for is_diag in diagonal_flags]
    print(
        f"original layer shape: {original_shape_tuple} "
        f"merged shape: {merged_shape_tuple} "
        f"preconditioners: {preconditioner_types}"
    )
