# PSGD-QUAD

`pip install quad-torch`

An implementation of PSGD-QUAD for PyTorch.

```python
import torch
from quad_torch import QUAD

model = torch.nn.Linear(10, 10)
optimizer = QUAD(
    model.parameters(),
    lr=0.001,
    lr_style="adam",  # "adam", "mu-p", or None
    momentum=0.95,
    weight_decay=0.1,
    preconditioner_lr=0.6,
    max_size_dense=8192,
    max_skew_dense=1.0,
    noise_scale=1e-8,
    normalize_grads=False,
    dtype=torch.bfloat16,
)
```

`lr_style` can be "adam" for adam-style scaling, "mu-p" for mu-p scaling based on sqrt(G.shape[-2]), or None for 
PSGD scaling of RMS=1.0.


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

2024 Evan Walters, Omead Pooladzandi, Xi-Lin Li


[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
