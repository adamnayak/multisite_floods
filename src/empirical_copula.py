from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Callable, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, kstest
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid, trapezoid

from marginals_pos import fit_with_auto_shift 


# ---------------------------
# Default marginal: "log-spline-like" 1D density
# ---------------------------

@dataclass
class LogSplineMarginal:
    """
    Practical Python surrogate for logspline density estimation:

    1) Estimate a smooth density on a grid using a Gaussian KDE-like approach
       (implemented via histogram smoothing in log space for robustness),
    2) Fit a spline to log(pdf) over that grid,
    3) Numerically integrate to get CDF, and invert via interpolation.

    Notes:
    - This is *not* the exact R `logspline` algorithm, but it behaves similarly:
      smooth, flexible, nonparametric density with stable sampling/ppf.
    - If you already have a preferred univariate fit (parametric or nonparam),
      pass it via the simulator's `marginal_factory`.
    """
    grid_size: int = 512
    smooth: float = 1e-2  # spline smoothing factor
    clip_quantiles: Tuple[float, float] = (0.001, 0.999)
    eps: float = 1e-12

    # fitted fields
    x_grid: Optional[np.ndarray] = None
    logpdf_spline: Optional[UnivariateSpline] = None
    cdf_grid: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "LogSplineMarginal":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            raise ValueError("Need at least 5 finite samples to fit a marginal.")

        lo = np.quantile(x, self.clip_quantiles[0])
        hi = np.quantile(x, self.clip_quantiles[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.min(x), np.max(x)
            if hi <= lo:
                hi = lo + 1.0

        # grid
        self.x_grid = np.linspace(lo, hi, self.grid_size)

        # density estimate via histogram -> smooth -> log spline
        hist, edges = np.histogram(x, bins=max(30, int(np.sqrt(x.size))), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # interpolate hist to grid, avoid zeros
        pdf0 = np.interp(self.x_grid, centers, hist, left=hist[0], right=hist[-1])
        pdf0 = np.maximum(pdf0, self.eps)

        # fit spline to log pdf
        self.logpdf_spline = UnivariateSpline(self.x_grid, np.log(pdf0), s=self.smooth * self.grid_size)

        # build normalized pdf + cdf on grid
        logpdf = self.logpdf_spline(self.x_grid)
        pdf = np.exp(logpdf)
        pdf = np.maximum(pdf, self.eps)

        # normalize
        area = trapezoid(pdf, self.x_grid)
        pdf = pdf / max(area, self.eps)

        cdf = cumulative_trapezoid(pdf, self.x_grid, initial=0.0)
        cdf = np.clip(cdf, 0.0, 1.0)
        # enforce last exactly 1 for stable inversion
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        self.cdf_grid = cdf
        return self

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.logpdf_spline is None:
            raise RuntimeError("Marginal not fit.")
        x = np.asarray(x, dtype=float)
        return np.exp(self.logpdf_spline(x))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.x_grid is None or self.cdf_grid is None:
            raise RuntimeError("Marginal not fit.")
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.x_grid, self.cdf_grid, left=0.0, right=1.0)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        if self.x_grid is None or self.cdf_grid is None:
            raise RuntimeError("Marginal not fit.")
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        return np.interp(u, self.cdf_grid, self.x_grid)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        u = rng.random(n)
        return self.ppf(u)


# ---------------------------
# Empirical copula simulator
# ---------------------------

class EmpiricalCopulaSimulator:
    """
    Empirical copula simulator for an (n_samples x n_vars) dataset.

    Capabilities:
      0) Preprocess and store metadata (columns, n, d)
      1) Fit rank-based empirical copula on original data (rank matrix)
      2) Fit marginals (default LogSplineMarginal) or user-provided,
         optionally run KS tests, sample Monte Carlo marginals,
         then impose dependence via rank reordering
      3) Plot marginals (raw vs simulated)

    The dependence-imposition step follows the rank reordering logic
    described in Lall et al. (2016): bootstrap rank vectors and map
    ranks into sorted marginal samples (their Eq. 7 / algorithm steps).
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None,
        marginal_factory: Optional[Callable[[], Any]] = None,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)

        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.columns = list(self.df.columns) if columns is None else list(columns)
            self.X = self.df[self.columns].to_numpy(dtype=float)
            self.index = self.df.index
        else:
            X = np.asarray(data, dtype=float)
            if X.ndim != 2:
                raise ValueError("data must be 2D (n_samples x n_vars).")
            self.X = X
            self.columns = columns if columns is not None else [f"var_{i}" for i in range(X.shape[1])]
            self.index = pd.RangeIndex(X.shape[0])
            self.df = pd.DataFrame(self.X, columns=self.columns, index=self.index)

        self.n, self.d = self.X.shape
        if len(self.columns) != self.d:
            raise ValueError("columns length must match number of variables (data.shape[1]).")

        # default marginal factory
        self.marginal_factory = marginal_factory or (lambda: LogSplineMarginal())

        # fitted objects
        self.rank_mat_: Optional[np.ndarray] = None
        self.marginals_: Dict[str, Any] = {}
        self.ks_results_: Optional[pd.DataFrame] = None

    # ---------- 1) empirical copula fit ----------

    def fit_empirical_copula(self, method: str = "ordinal", smooth_ranks: bool = False) -> "EmpiricalCopulaSimulator":
        """
        Fit the empirical copula as a rank matrix.

        Parameters
        ----------
        method : str
            scipy.stats.rankdata method, e.g., "ordinal" (no ties averaging).
        smooth_ranks : bool
            If True, apply a "smoothed" rank approach: jitter each rank into
            a small interval (plotting-position uncertainty), then re-rank.
            Motivated by the paper's discussion of treating empirical CDF
            values as uncertain rather than point values.
        """
        U = np.empty((self.n, self.d), dtype=float)
        R = np.empty((self.n, self.d), dtype=int)
    
        for j, col in enumerate(self.columns):
            r = rankdata(self.df[col].to_numpy(), method=method).astype(int)  # 1..n
    
            # always store integer ranks (0-based)
            R[:, j] = r - 1
    
            # store U either as plotting positions or smoothed within rank bins
            if smooth_ranks:
                u_lo = (r - 0.5) / (self.n + 1.0)
                u_hi = (r + 0.5) / (self.n + 1.0)
                U[:, j] = u_lo + (u_hi - u_lo) * self.rng.random(self.n)
            else:
                U[:, j] = r / (self.n + 1.0)
    
        self.u_mat_ = U
        self.rank_mat_ = R
        return self

    # ---------- 2) marginals fit + KS ----------

    def fit_marginals(self, marginals=None, run_ks=False, auto_shift_positive=True) -> "EmpiricalCopulaSimulator":
        """
        Fit univariate distributions for each variable.
    
        If auto_shift_positive=True and a marginal declares supports_nonnegative=True,
        negative-valued data are shifted to positive for fitting and then shifted back
        automatically for cdf/ppf/rvs via ShiftToPositive wrapper.
        """
        self.marginals_.clear()
        self.transform_report_ = []
    
        for col in self.columns:
            m = (marginals[col] if (marginals is not None and col in marginals) else self.marginal_factory())
            x = self.df[col].to_numpy(dtype=float)
    
            m_fitted, rep = fit_with_auto_shift(m, x, enable=auto_shift_positive)
            self.marginals_[col] = m_fitted
    
            if rep is not None:
                self.transform_report_.append({"variable": col, **rep})
    
        if self.transform_report_:
            self.transform_report_ = pd.DataFrame(self.transform_report_)
            print("Auto shift-to-positive applied (fit shifted; outputs returned on original scale):")
            print(self.transform_report_)
    
        if run_ks:
            rows = []
            for col in self.columns:
                x = self.df[col].to_numpy(dtype=float)
                m = self.marginals_[col]
                stat, pval = kstest(x[np.isfinite(x)], cdf=lambda v: m.cdf(np.asarray(v)))
                rows.append({"variable": col, "ks_stat": stat, "p_value": pval})
            self.ks_results_ = pd.DataFrame(rows).set_index("variable")
    
        return self

    # ---------- sampling + dependence imposition ----------

    def sample(
        self,
        n_samples: Optional[int] = None,
        bootstrap: bool = True,
        enforce_nonneg_int_cols: Optional[List[str]] = None,
        use_u: bool = True,
    ) -> pd.DataFrame:
        if self.rank_mat_ is None:
            raise RuntimeError("Call fit_empirical_copula() first.")
        if not self.marginals_:
            raise RuntimeError("Call fit_marginals() first.")
    
        n_out = int(n_samples) if n_samples is not None else self.n
    
        # 1) draw independent marginal samples, then sort each column
        R_prime = np.empty((n_out, self.d), dtype=float)
        for j, col in enumerate(self.columns):
            m = self.marginals_[col]
            x_samp = m.rvs(n_out, random_state=self.rng) if hasattr(m, "rvs") else m.ppf(self.rng.random(n_out))
            R_prime[:, j] = np.sort(np.asarray(x_samp, dtype=float))
    
        # 2) choose rank/U rows
        if bootstrap:
            idx = self.rng.integers(0, self.n, size=n_out)
        else:
            # no resampling: cycle through original rows to reach n_out
            idx = np.arange(n_out) % self.n
    
        # 3) map U (preferred) or ranks -> indices in [0, n_out-1]
        # idx: bootstrapped row indices length n_out
        z_prime = self.rank_mat_[idx, :]  # (n_out, d), integers 0..n-1
        
        if use_u:
            # Fresh jitter per simulated draw (key!)
            # Put each pseudo-observation uniformly within its rank bin
            jitter = self.rng.random(size=z_prime.shape)  # in [0, 1)
            U_boot = (z_prime.astype(float) + jitter) / float(self.n)  # in [0,1)
            U_boot = np.clip(U_boot, 1e-12, 1 - 1e-12)
        
            z_idx = np.floor(U_boot * n_out).astype(int)
            z_idx = np.clip(z_idx, 0, n_out - 1)
        else:
            u = (z_prime.astype(float) + 0.5) / float(self.n)
            z_idx = np.floor(u * n_out).astype(int)
            z_idx = np.clip(z_idx, 0, n_out - 1)
    
        # 4) map indices -> sorted marginals
        W = np.empty((n_out, self.d), dtype=float)
        for j in range(self.d):
            W[:, j] = R_prime[z_idx[:, j], j]
    
        out = pd.DataFrame(W, columns=self.columns)
    
        # optional post-processing for discrete nonnegative variables
        if enforce_nonneg_int_cols:
            for c in enforce_nonneg_int_cols:
                if c in out.columns:
                    v = np.round(out[c].to_numpy())
                    out[c] = np.where(v < 0, 0, v).astype(int)
    
        return out
    
    
    # ---------- 3) plotting ----------

    def plot_marginals(
        self,
        simulated: Optional[pd.DataFrame] = None,
        which: str = "both",  # "raw" | "sim" | "both"
        bins: int = 40,
        density: bool = True,
        max_cols: int = 4,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plot marginal distributions for each variable.
        """
        cols = self.columns
        n = len(cols)
        ncols = min(max_cols, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        for k, col in enumerate(cols):
            ax = axes[k]
            if which in ("raw", "both"):
                ax.hist(self.df[col].to_numpy(), bins=bins, density=density, alpha=0.5, label="raw")
            if which in ("sim", "both"):
                if simulated is None:
                    raise ValueError("Provide `simulated` when which includes 'sim'.")
                ax.hist(simulated[col].to_numpy(), bins=bins, density=density, alpha=0.5, label="sim")

            ax.set_title(col)
            ax.legend()

        # hide unused axes
        for k in range(n, len(axes)):
            axes[k].axis("off")

        plt.tight_layout()
        plt.show()