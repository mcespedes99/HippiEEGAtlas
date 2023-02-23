"""Utility functions."""
from .base import mldivide, mrdivide
from .covariances import (block_covariance, convmtx, cov_lags,
                          nonlinear_eigenspace, pca, regcov, tscov, tsxcov)
from .matrix import (fold, matmul3d, multishift, multismooth, normcol,
                     relshift, shift, shiftnd, sliding_window, theshapeof,
                     unfold, unsqueeze, widen_mask)
