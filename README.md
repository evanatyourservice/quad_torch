# PSGD-QUAD and PSGD-Procrustes

`pip install quad-torch`

Implementations of PSGD-Procrustes and PSGD-QUAD for PyTorch.

Procrustes is a more exact implementation that instead updates Q using Q0.5EQ1.5.

```python
import torch
from quad_torch import Procrustes

model = torch.nn.Linear(10, 10)
optimizer = Procrustes(
    model.parameters(),
    lr=0.001,
    lr_style="adam",
    momentum=0.95,
    weight_decay=0.1,
    max_size_dense=16384,
    max_skew_dense=1.0,
    preconditioner_lr=0.7,
    preconditioner_init_scale=None,
    noise_scale=1e-9,
    dtype=torch.bfloat16,
)
```

### **A few notes:**

**LAYER RESHAPING CAVEAT**: This is a slightly simplified version that, instead of generalizing to any-dimensional layers, reshapes layers with greater than 2 dimensions into the most square matrix, then performs preconditioning. For example, a layer shaped [32, 64, 1024] will be reshaped into a matrix shaped [2048, 1024], then preconditioned. *`max_size_dense` and `max_skew_dense` apply to the **merged** matrix shape, not the original layer shape.*

**LR STYLE**: `lr_style="adam"` is the default and scales the update to match adam's behavior for LR and weight decay. PSGD's raw update aims for RMS=1.0 (`lr_style=None`).

**LR WARMUP**: Don't have to use as much LR warmup, usually either 0 or 100 steps

**MAX_SKEW_DENSE**: `max_skew_dense` default is 1.0, which makes dimensions with skew larger than 1.0 have diagonal preconditioners, but you can set to float('inf') to make all preconditioners dense. For example, `max_skew_dense=1.0` behavior: [128, 128]=[dense, dense], [128, 1024]=[dense, diagonal], but `max_skew_dense=float('inf')` behavior: [128, 128]=[dense, dense], [128, 1024]=[dense, dense]. 1D layers always have a diagonal preconditioner.

**DTYPE**: `dtype=torch.bfloat16` should be fine for most problems, but if a problem is particularly sensitive then you can try `None` to default to gradient dtypes or `torch.float32` to force global f32 precision.


## Resources

Xi-Lin Li's repo: https://github.com/lixilinx/psgd_torch

PSGD papers and resources listed from Xi-Lin's repo

1) Xi-Lin Li. Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting losses and Kronecker product preconditioners.)
2) Xi-Lin Li. Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)
3) Xi-Lin Li. Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. See [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Xi-Lin Li. Stochastic Hessian fittings on Lie groups, [arXiv:2402.11858](https://arxiv.org/abs/2402.11858), 2024. (Some theoretical works on the efficiency of PSGD. The Hessian fitting problem is shown to be strongly convex on set ${\rm GL}(n, \mathbb{R})/R_{\rm polar}$.)
5) Omead Pooladzandi, Xi-Lin Li. Curvature-informed SGD via general purpose Lie-group preconditioners, [arXiv:2402.04553](https://arxiv.org/abs/2402.04553), 2024. (Plenty of benchmark results and analyses for PSGD vs. other optimizers.)


## License

[![CC BY 4.0][cc-by-image]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

2025 Xi-Lin Li, Omead Pooladzandi, Evan Walters, Lucas Nestler


[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
