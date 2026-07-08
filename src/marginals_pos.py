# marginals_pos.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, ClassVar

import numpy as np

from scipy.stats import (
    poisson,
    nbinom,
    expon,
    gamma as gamma_dist,
    lognorm,
    weibull_min,
    genextreme,
    kstest,
)
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid, trapezoid


# ----------------------------
# Protocol / interface
# ----------------------------

class Marginal(Protocol):
    def fit(self, x: np.ndarray) -> "Marginal": ...
    def cdf(self, x: np.ndarray) -> np.ndarray: ...
    def ppf(self, u: np.ndarray) -> np.ndarray: ...
    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray: ...


def _as_float_1d(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return x


# ----------------------------
# Discrete marginals
# ----------------------------

@dataclass
class TruncatedLognormalMarginal:
    """
    Left-truncated Lognormal at threshold a implemented as a shift:

        X | X >= a  ~  a + Y,   Y ~ Lognormal(mu, sigma)

    So support is [a, ∞) exactly (no additional renormalization).

    MLE (on shifted positives y = X - a):
        mu    = mean(log(y))
        sigma = std(log(y))

    Threshold handling mirrors TruncatedExponentialMarginal:

    If auto_threshold=True and threshold is None, infer a from data:
      - default: a = min(x)
      - if use_min_positive_if_zeros=True: a = min(x[x>0]) when zeros present
        (useful when data have many structural zeros and positives are thresholded)

    Notes:
    - Negatives are clipped to 0.0 before threshold inference (consistent with your other marginals).
    - For numerical stability, we enforce y >= 1e-12 before taking logs.
    """
    supports_nonnegative: ClassVar[bool] = True

    threshold: Optional[float] = None  # if None and auto_threshold, inferred in fit()
    mu_: float = 0.0
    sigma_: float = 1.0

    auto_threshold: bool = True
    use_min_positive_if_zeros: bool = True

    # diagnostics / bookkeeping
    threshold_inferred_: Optional[float] = None
    used_min_positive_: bool = False

    def fit(self, x: np.ndarray) -> "TruncatedLognormalMarginal":
        # assumes you have the same helper as before
        x = _as_float_1d(x)

        # Nonnegative support: clip negatives
        x = np.clip(x, 0.0, None)

        if x.size == 0:
            # fall back to a benign default
            self.mu_ = 0.0
            self.sigma_ = 1.0
            self.threshold_inferred_ = None
            self.used_min_positive_ = False
            return self

        # --- Infer threshold if requested ---
        if self.auto_threshold and self.threshold is None:
            self.used_min_positive_ = False

            if self.use_min_positive_if_zeros and np.any(x == 0.0):
                xp = x[x > 0.0]
                if xp.size > 0:
                    a = float(np.min(xp))
                    self.used_min_positive_ = True
                else:
                    # all zeros: degenerate; treat a=0
                    a = 0.0
            else:
                a = float(np.min(x))

            self.threshold = a
            self.threshold_inferred_ = a
        else:
            # user-provided threshold (or None -> treat as 0.0)
            a = float(self.threshold) if self.threshold is not None else 0.0
            self.threshold = a
            self.threshold_inferred_ = None
            self.used_min_positive_ = False

        a = float(self.threshold)

        # keep only data consistent with truncation
        x = x[x >= a]
        if x.size == 0:
            # no data above threshold; keep defaults
            self.mu_ = 0.0
            self.sigma_ = 1.0
            return self

        # Shift to lognormal on (0, ∞)
        y = x - a

        # Guard against y<=0 due to rounding or exact threshold hits
        eps = 1e-12
        y = np.clip(y, eps, None)

        logy = np.log(y)
        self.mu_ = float(logy.mean())
        # population std (ddof=0) to match usual lognormal parameterization
        self.sigma_ = max(float(logy.std(ddof=0)), 1e-6)
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        CDF of X with support [a, ∞):

          F(x) = 0                                       for x < a
               = F_Y(x - a)                             for x >= a

        where Y ~ Lognormal(mu_, sigma_).
        """
        x = np.asarray(x, dtype=float)
        a = float(self.threshold or 0.0)

        out = np.zeros_like(x, dtype=float)
        mask = x >= a
        if not np.any(mask):
            return out

        y = x[mask] - a
        # y>0 is guaranteed by construction, but guard anyway
        y = np.clip(y, 0.0, None)
        # lognorm parameterization: s = sigma, scale = exp(mu)
        out[mask] = lognorm.cdf(y, s=self.sigma_, scale=np.exp(self.mu_))
        return out

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """
        Quantile function on [a, ∞):

          Q(u) = a + Q_Y(u),   Y ~ Lognormal(mu_, sigma_)
        """
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        a = float(self.threshold or 0.0)

        yq = lognorm.ppf(u, s=self.sigma_, scale=np.exp(self.mu_))
        return a + yq

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Draw n samples from X = a + Y, Y ~ Lognormal(mu_, sigma_).
        """
        rng = random_state if random_state is not None else np.random.default_rng()
        a = float(self.threshold or 0.0)

        y = lognorm.rvs(
            s=self.sigma_,
            scale=np.exp(self.mu_),
            size=int(n),
            random_state=rng,
        )
        return a + y

@dataclass
class TruncatedExponentialMarginal:
    """
    Left-truncated Exponential at threshold a:

        X | X >= a  ~  a + Y,     Y ~ Exponential(scale)

    MLE for scale given truncation at a:
        scale = mean(X - a)  over observed X >= a

    If auto_threshold=True and threshold is None, infer a from data:
      - default: a = min(x)
      - if use_min_positive_if_zeros=True: a = min(x[x>0]) when zeros present
        (useful when data have many structural zeros and positives are thresholded)
    """
    supports_nonnegative: ClassVar[bool] = True

    threshold: Optional[float] = None  # if None and auto_threshold, inferred in fit()
    scale_: float = 1.0

    auto_threshold: bool = True
    use_min_positive_if_zeros: bool = True  # recommended default for “thresholded positives + zeros”

    # diagnostics / bookkeeping
    threshold_inferred_: Optional[float] = None
    used_min_positive_: bool = False

    def fit(self, x: np.ndarray) -> "TruncatedExponentialMarginal":
        x = _as_float_1d(x)

        # Nonnegative support: clip negatives (or you could raise; clipping is consistent with your other marginals)
        x = np.clip(x, 0.0, None)

        if x.size == 0:
            self.scale_ = 1.0
            self.threshold_inferred_ = None
            self.used_min_positive_ = False
            return self

        # --- Infer threshold if requested ---
        if self.auto_threshold and self.threshold is None:
            self.used_min_positive_ = False

            if self.use_min_positive_if_zeros and np.any(x == 0.0):
                xp = x[x > 0.0]
                if xp.size > 0:
                    a = float(np.min(xp))
                    self.used_min_positive_ = True
                else:
                    # all zeros: degenerate
                    a = 0.0
            else:
                a = float(np.min(x))

            self.threshold = a
            self.threshold_inferred_ = a
        else:
            # user-provided threshold (or default None -> treat as 0.0)
            a = float(self.threshold) if self.threshold is not None else 0.0
            self.threshold = a
            self.threshold_inferred_ = None
            self.used_min_positive_ = False

        a = float(self.threshold)

        # keep only data consistent with truncation (and for exponential we want y>0 for stable MLE)
        x = x[x >= a]
        if x.size == 0:
            self.scale_ = 1.0
            return self

        y = x - a

        # If many points equal exactly a (e.g., rounding), mean(y) can be ~0; guard it
        m = float(np.mean(y)) if y.size else 0.0
        self.scale_ = max(m, 1e-12)
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Conditional CDF of X | X>=a:
          F(x) = 0                         for x < a
               = 1 - exp(-(x-a)/scale)     for x >= a
        """
        x = np.asarray(x, dtype=float)
        a = float(self.threshold or 0.0)

        out = np.zeros_like(x, dtype=float)
        mask = x >= a
        out[mask] = expon.cdf(x[mask] - a, loc=0.0, scale=self.scale_)
        return out

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        a = float(self.threshold or 0.0)
        return a + expon.ppf(u, loc=0.0, scale=self.scale_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        a = float(self.threshold or 0.0)
        y = expon.rvs(loc=0.0, scale=self.scale_, size=int(n), random_state=rng)
        return a + y

@dataclass
class PoissonMarginal:
    supports_nonnegative: ClassVar[bool] = True
    """Poisson with MLE lambda = mean(x)."""
    lam_: float = 1.0

    def fit(self, x: np.ndarray) -> "PoissonMarginal":
        x = _as_float_1d(x)
        if x.size == 0:
            self.lam_ = 0.0
            return self
        # Ensure nonnegative integer-ish
        x = np.clip(x, 0, None)
        self.lam_ = float(np.mean(x))
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return poisson.cdf(x, mu=self.lam_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return poisson.ppf(u, mu=self.lam_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        return poisson.rvs(mu=self.lam_, size=n, random_state=rng)


@dataclass
class NegBinMarginal:
    supports_nonnegative: ClassVar[bool] = True
    """
    Negative Binomial for overdispersed counts.

    Uses scipy.stats.nbinom parameterization:
      nbinom(n, p) with mean = n*(1-p)/p

    Fit strategy:
      Method of moments. scipy's nbinom is an rv_discrete with no .fit(), so a
      scipy MLE is unavailable; MoM gives the standard NB estimator
      (n = mean^2 / (var - mean), p = mean / var), falling back to a
      near-Poisson limit when the data are not overdispersed.
    """
    n_: float = 1.0
    p_: float = 0.5

    def fit(self, x: np.ndarray) -> "NegBinMarginal":
        x = _as_float_1d(x)
        if x.size == 0:
            self.n_, self.p_ = 1.0, 0.5
            return self
        x = np.clip(np.round(x), 0, None)

        # Method of moments (scipy's discrete nbinom has no .fit()).
        m = float(np.mean(x))
        v = float(np.var(x, ddof=0))
        if v <= m + 1e-12:
            # Not overdispersed; approximate with large n (close to Poisson)
            self.n_ = 1e6
            self.p_ = self.n_ / (self.n_ + m + 1e-12)
            return self

        # For NB: var = mean + mean^2 / n  => n = mean^2 / (var - mean)
        n_hat = (m * m) / max(v - m, 1e-12)
        p_hat = n_hat / (n_hat + m)
        self.n_ = float(max(n_hat, 1e-6))
        self.p_ = float(np.clip(p_hat, 1e-9, 1 - 1e-9))
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return nbinom.cdf(x, n=self.n_, p=self.p_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return nbinom.ppf(u, n=self.n_, p=self.p_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        return nbinom.rvs(n=self.n_, p=self.p_, size=n, random_state=rng)


# ----------------------------
# Continuous positive marginals
# ----------------------------

@dataclass
class ExponentialMarginal:
    supports_nonnegative: ClassVar[bool] = True
    """Exponential with loc=0; MLE scale = mean(x)."""
    scale_: float = 1.0

    def fit(self, x: np.ndarray) -> "ExponentialMarginal":
        x = _as_float_1d(x)
        x = np.clip(x, 0.0, None)
        self.scale_ = float(np.mean(x)) if x.size else 1.0
        self.scale_ = max(self.scale_, 1e-12)
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return expon.cdf(x, loc=0.0, scale=self.scale_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return expon.ppf(u, loc=0.0, scale=self.scale_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        return expon.rvs(loc=0.0, scale=self.scale_, size=n, random_state=rng)


@dataclass
class GammaMarginal:
    supports_nonnegative: ClassVar[bool] = True
    """
    Gamma with loc fixed at 0 (positive support).
    If zeros are present, replace them with a small epsilon so scipy's MLE won't fail.
    """
    a_: float = 1.0
    scale_: float = 1.0
    zero_policy: str = "epsilon"   # "epsilon" | "drop"
    eps: float = 1e-6              # relative to data scale if possible

    def fit(self, x: np.ndarray) -> "GammaMarginal":
        x = _as_float_1d(x)
        x = np.clip(x, 0.0, None)

        if x.size < 5 or np.allclose(x, 0.0):
            self.a_, self.scale_ = 1.0, 1.0
            return self

        if self.zero_policy == "drop":
            xp = x[x > 0]
            if xp.size < 5:
                self.a_, self.scale_ = 1.0, max(float(np.mean(xp)) if xp.size else 1.0, 1e-12)
                return self
            x_use = xp

        elif self.zero_policy == "epsilon":
            # choose epsilon relative to the scale of positive data if available
            xp = x[x > 0]
            if xp.size > 0:
                # e.g., 1e-6 * median positive (or min positive if tiny)
                base = float(np.median(xp))
                base = base if base > 0 else float(np.min(xp))
                eps = max(self.eps * base, 1e-12)
            else:
                eps = 1e-12
            x_use = x.copy()
            x_use[x_use <= 0] = eps

        else:
            raise ValueError("zero_policy must be 'epsilon' or 'drop'")

        a, loc, scale = gamma_dist.fit(x_use, floc=0.0)
        self.a_ = float(max(a, 1e-6))
        self.scale_ = float(max(scale, 1e-12))
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gamma_dist.cdf(x, a=self.a_, loc=0.0, scale=self.scale_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return gamma_dist.ppf(u, a=self.a_, loc=0.0, scale=self.scale_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        return gamma_dist.rvs(a=self.a_, loc=0.0, scale=self.scale_, size=n, random_state=rng)


@dataclass
class LogNormalMarginal:
    supports_nonnegative: ClassVar[bool] = True
    """
    Lognormal with loc fixed at 0.
    Fit via scipy MLE: lognorm.fit(x, floc=0) -> (s, loc, scale)
    """
    s_: float = 1.0
    scale_: float = 1.0

    def fit(self, x: np.ndarray) -> "LogNormalMarginal":
        x = _as_float_1d(x)
        x = np.clip(x, 0.0, None)
        xp = x[x > 0]
        if xp.size < 5:
            self.s_, self.scale_ = 1.0, max(float(np.mean(xp)) if xp.size else 1.0, 1e-12)
            return self
        s, loc, scale = lognorm.fit(xp, floc=0.0)
        self.s_ = float(max(s, 1e-6))
        self.scale_ = float(max(scale, 1e-12))
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return lognorm.cdf(x, s=self.s_, loc=0.0, scale=self.scale_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return lognorm.ppf(u, s=self.s_, loc=0.0, scale=self.scale_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        return lognorm.rvs(s=self.s_, loc=0.0, scale=self.scale_, size=n, random_state=rng)


@dataclass
class WeibullMarginal:
    supports_nonnegative: ClassVar[bool] = True
    """
    Weibull (weibull_min) with loc fixed at 0.
    Fit via scipy MLE: weibull_min.fit(x, floc=0) -> (c, loc, scale)
    """
    c_: float = 1.5
    scale_: float = 1.0

    def fit(self, x: np.ndarray) -> "WeibullMarginal":
        x = _as_float_1d(x)
        x = np.clip(x, 0.0, None)
        xp = x[x > 0]
        if xp.size < 5:
            self.c_, self.scale_ = 1.5, max(float(np.mean(xp)) if xp.size else 1.0, 1e-12)
            return self
        c, loc, scale = weibull_min.fit(xp, floc=0.0)
        self.c_ = float(max(c, 1e-6))
        self.scale_ = float(max(scale, 1e-12))
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return weibull_min.cdf(x, c=self.c_, loc=0.0, scale=self.scale_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return weibull_min.ppf(u, c=self.c_, loc=0.0, scale=self.scale_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        return weibull_min.rvs(c=self.c_, loc=0.0, scale=self.scale_, size=n, random_state=rng)


# ----------------------------
# Extremes / tail (GEV)
# ----------------------------

@dataclass
class GEVMarginal:
    supports_nonnegative: ClassVar[bool] = False
    """
    Generalized Extreme Value (GEV) using scipy.stats.genextreme.

    scipy's genextreme shape parameter `c` corresponds to -xi in some conventions.
    We keep scipy's parameterization to avoid confusion.
    """
    c_: float = 0.0
    loc_: float = 0.0
    scale_: float = 1.0

    def fit(self, x: np.ndarray) -> "GEVMarginal":
        x = _as_float_1d(x)
        if x.size < 10:
            # fallback: roughly centered
            self.c_, self.loc_, self.scale_ = 0.0, float(np.mean(x)) if x.size else 0.0, float(np.std(x) + 1e-12)
            return self
        c, loc, scale = genextreme.fit(x)
        self.c_ = float(c)
        self.loc_ = float(loc)
        self.scale_ = float(max(scale, 1e-12))
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return genextreme.cdf(x, c=self.c_, loc=self.loc_, scale=self.scale_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return genextreme.ppf(u, c=self.c_, loc=self.loc_, scale=self.scale_)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        return genextreme.rvs(c=self.c_, loc=self.loc_, scale=self.scale_, size=n, random_state=rng)


# ----------------------------
# Logspline-like (nonparametric) marginal
# ----------------------------

@dataclass
class LogSplineMarginal:
    supports_nonnegative: ClassVar[bool] = False
    """
    Practical "logspline-like" nonparametric 1D marginal:
      - Fit smooth spline to log(pdf) on a grid (constructed from histogram density),
      - Numerical integration -> CDF,
      - Inversion via interpolation -> PPF and sampling.

    This is a robust baseline; it is not the exact R `logspline` implementation,
    but behaves similarly as a smooth density estimator with stable sampling.
    """
    grid_size: int = 512
    smooth: float = 1e-2
    clip_quantiles: Tuple[float, float] = (0.001, 0.999)
    eps: float = 1e-12

    x_grid: Optional[np.ndarray] = None
    logpdf_spline: Optional[UnivariateSpline] = None
    cdf_grid: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "LogSplineMarginal":
        x = _as_float_1d(x)
        if x.size < 5:
            raise ValueError("Need at least 5 finite samples to fit LogSplineMarginal.")

        lo = np.quantile(x, self.clip_quantiles[0])
        hi = np.quantile(x, self.clip_quantiles[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi <= lo:
                hi = lo + 1.0

        self.x_grid = np.linspace(lo, hi, self.grid_size)

        # histogram density
        nbins = max(30, int(np.sqrt(x.size)))
        hist, edges = np.histogram(x, bins=nbins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        pdf0 = np.interp(self.x_grid, centers, hist, left=hist[0], right=hist[-1])
        pdf0 = np.maximum(pdf0, self.eps)

        self.logpdf_spline = UnivariateSpline(
            self.x_grid, np.log(pdf0), s=self.smooth * self.grid_size
        )

        logpdf = self.logpdf_spline(self.x_grid)
        pdf = np.exp(logpdf)
        pdf = np.maximum(pdf, self.eps)

        area = float(trapezoid(pdf, self.x_grid))
        pdf = pdf / max(area, self.eps)

        cdf = cumulative_trapezoid(pdf, self.x_grid, initial=0.0)
        cdf = np.clip(cdf, 0.0, 1.0)
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        self.cdf_grid = cdf
        return self

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.logpdf_spline is None:
            raise RuntimeError("LogSplineMarginal not fit.")
        return np.exp(self.logpdf_spline(np.asarray(x, dtype=float)))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.x_grid is None or self.cdf_grid is None:
            raise RuntimeError("LogSplineMarginal not fit.")
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.x_grid, self.cdf_grid, left=0.0, right=1.0)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        if self.x_grid is None or self.cdf_grid is None:
            raise RuntimeError("LogSplineMarginal not fit.")
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        return np.interp(u, self.cdf_grid, self.x_grid)

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = random_state if random_state is not None else np.random.default_rng()
        u = rng.random(n)
        return self.ppf(u)



# ----------------------------
# Auto shift-to-positive wrapper (for positive-support marginals)
# ----------------------------

@dataclass
class ShiftToPositive:
    """Wrap a nonnegative-support marginal to handle negative observed values.

    If the fitted marginal declares supports_nonnegative=True and the data contain
    values < 0, we shift data by -min(x) during fit so the support becomes >= 0.
    All modeled values returned by cdf/ppf/rvs are shifted back to the original
    scale automatically.

    This preserves information in the negative tail without clipping, while still
    allowing positive-support distributions (gamma, weibull, lognormal, etc.) to fit.
    """
    base: Marginal
    shift_: float = 0.0
    applied_: bool = False

    def fit(self, x: np.ndarray) -> "ShiftToPositive":
        x = _as_float_1d(x)
        if x.size == 0:
            self.shift_ = 0.0
            self.applied_ = False
            self.base.fit(x)
            return self

        mn = float(np.min(x))
        if mn < 0:
            self.shift_ = -mn
            self.applied_ = True
            self.base.fit(x + self.shift_)
        else:
            self.shift_ = 0.0
            self.applied_ = False
            self.base.fit(x)

        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.base.cdf(np.asarray(x, dtype=float) + self.shift_)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.base.ppf(np.asarray(u, dtype=float)) - self.shift_

    def rvs(self, n: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        return self.base.rvs(n, random_state=random_state) - self.shift_

    def report(self) -> Dict[str, Any]:
        return {"shift": float(self.shift_), "applied": bool(self.applied_), "base": type(self.base).__name__}


def fit_with_auto_shift(
    marginal: Marginal,
    x: np.ndarray,
    enable: bool = True,
) -> Tuple[Marginal, Optional[Dict[str, Any]]]:
    """Fit a marginal with automatic shift-to-positive when appropriate.

    If enable=True and:
      - marginal.supports_nonnegative is True, and
      - min(x) < 0
    then we fit using ShiftToPositive(marginal). Outputs are automatically shifted
    back to the original scale via the wrapper.

    Returns
    -------
    (marginal_fitted, report)
        report is None if no shift was applied; otherwise includes shift details.
    """
    x = np.asarray(x, dtype=float)
    xfinite = x[np.isfinite(x)]

    if not enable or xfinite.size == 0:
        marginal.fit(x)
        return marginal, None

    supports_nonneg = bool(getattr(marginal, "supports_nonnegative", False))
    if supports_nonneg and float(np.min(xfinite)) < 0:
        wrapper = ShiftToPositive(base=marginal)
        wrapper.fit(x)
        return wrapper, wrapper.report()

    marginal.fit(x)
    return marginal, None

# ----------------------------
# Registry + spec parser
# ----------------------------

MarginalSpec = Union[str, Dict[str, Any], Marginal, Callable[[], Marginal]]

_REGISTRY: Dict[str, Callable[..., Marginal]] = {
    "poisson": lambda **kw: PoissonMarginal(**kw),
    "negbin": lambda **kw: NegBinMarginal(**kw),
    "exponential": lambda **kw: ExponentialMarginal(**kw),
    "exponential_t": lambda **kw: TruncatedExponentialMarginal(**kw),
    "gamma": lambda **kw: GammaMarginal(**kw),
    "lognormal": lambda **kw: LogNormalMarginal(**kw),
    "lognormal_t": lambda **kw: TruncatedLognormalMarginal(**kw),
    "weibull": lambda **kw: WeibullMarginal(**kw),
    "gev": lambda **kw: GEVMarginal(**kw),
    "logspline": lambda **kw: LogSplineMarginal(**kw),
}


def make_marginal(spec: MarginalSpec) -> Marginal:
    """
    Create a marginal from a spec.

    Allowed specs:
      - string: "gamma", "gev", "logspline", ...
      - dict: {"name": "gamma", ...params}
      - Marginal object instance: used directly
      - callable factory: () -> Marginal
    """
    if isinstance(spec, str):
        name = spec.lower()
        if name not in _REGISTRY:
            raise KeyError(f"Unknown marginal '{spec}'. Available: {list(_REGISTRY)}")
        return _REGISTRY[name]()

    if callable(spec) and not isinstance(spec, dict):
        # factory
        m = spec()
        return m

    if isinstance(spec, dict):
        if "name" not in spec:
            raise ValueError("Dict marginal spec must include key 'name'.")
        name = str(spec["name"]).lower()
        if name not in _REGISTRY:
            raise KeyError(f"Unknown marginal '{name}'. Available: {list(_REGISTRY)}")
        params = {k: v for k, v in spec.items() if k != "name"}
        return _REGISTRY[name](**params)

    # assume already an object implementing the protocol
    return spec


def ks_test(x: np.ndarray, marginal: Marginal) -> Tuple[float, float]:
    """
    Convenience KS test wrapper. Note:
    - KS is formally exact for continuous distributions; for discrete it is approximate.
    """
    x = _as_float_1d(x)
    if x.size == 0:
        return np.nan, np.nan
    stat, p = kstest(x, cdf=lambda v: marginal.cdf(np.asarray(v)))
    return float(stat), float(p)