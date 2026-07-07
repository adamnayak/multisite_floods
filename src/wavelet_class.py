
"""
wavelet_class.py

Object-oriented wrapper for  wavelet utilities

Implements Morlet CWT, confidence intervals (red/white noise), reconstruction, and plotting.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, List, Union

import warnings
import numpy as np
import pandas as pd

from scipy.fftpack import fft, ifft
from scipy.stats import chi2
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

# seaborn is only used for the custom palette in the plot
import seaborn as sns


ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


@dataclass
class WaveletConfig:
    """Configuration knobs for the Morlet CWT implementation."""
    dt: float = 1.0                 # sampling interval (1 = "one time step"; for monthly series this can stay 1)
    dj: float = 0.025               # scale spacing
    morlet_wavenumber: float = 6.0  # "param" in Torrence & Compo (1998)
    pad_to_pow2: bool = True        # pad to next power of 2


class WaveletAnalyzer:
    """
    Wavelet analysis helper (Morlet CWT) with confidence intervals and reconstruction.

    Typical usage:
        wa = WaveletAnalyzer(WaveletConfig(dt=1, dj=0.025))
        wlt = wa.cwt(series)
        wlt_red = wa.confidence_interval(wlt, series, conf=0.95, noise_type="r")
        recon, sig_scales = wa.reconstruct(wlt_red, series)
    """

    def __init__(self, config: Optional[WaveletConfig] = None):
        self.config = config or WaveletConfig()

    # ----------------------------
    # Core transform
    # ----------------------------
    def cwt(self, y: ArrayLike) -> Dict[str, np.ndarray]:
        """
        Continuous Wavelet Transform (Morlet), returns dict with wave, period, power, coi, and average power
        """
        y = self._to_1d_array(y)
        dt = float(self.config.dt)
        dj = float(self.config.dj)
        pad = 1 if self.config.pad_to_pow2 else 0
        param = float(self.config.morlet_wavenumber)

        s0 = 2 * dt  # smallest resolvable scale
        n1 = len(y)
        J1 = int(np.floor((np.log2(n1 * dt / s0)) / dj))  # Eq. 10 (T&C 1998)

        # center and (optionally) pad to nearest power of 2
        x = y - np.mean(y)
                
        if pad == 1:
            base2 = int(np.trunc(np.log(n1) / np.log(2) + 0.4999))
            x = np.concatenate((x, np.zeros(2 ** (base2 + 1) - n1)))

        n = len(x)

        # wavenumber array
        k = np.arange(1, int(n / 2) + 1)
        k = k * ((2 * np.pi) / (n * dt))
        k = np.concatenate(([0], k, -np.flip(k[: int((n - 1) / 2)])))

        # FFT
        f = fft(x)

        # scales
        scale = s0 * 2 ** (np.arange(0, J1 + 1) * dj)  # Eq. 9
        wave = np.zeros((J1 + 1, n), dtype=complex)

        for a1 in range(J1 + 1):
            scl = scale[a1]
            expnt = np.zeros_like(k, dtype=float)
            pos = k > 0
            expnt[pos] = -((scl * k[pos] - param) ** 2) / 2.0
            norm = np.sqrt(scl * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
            daughter = norm * np.exp(expnt) * (k > 0)
            wave[a1, :] = ifft(f * daughter) #/ n

        fourier_factor = (4 * np.pi) / (param + np.sqrt(2 + param ** 2))
        period = fourier_factor * scale

        # cone of influence (COI)
        lengths = np.concatenate(
            (
                np.arange(1, np.floor((n1 + 1) / 2) + 1),
                np.arange(1, np.floor(n1 / 2) + 1)[::-1],
            )
        )
        coi = fourier_factor / np.sqrt(2) * lengths * dt

        # remove padding
        wave = wave[:, :n1]

        # normalized power spectrum
        power = np.abs(wave) ** 2 #np.var(wave)
        avg_power = np.mean(power, axis=1)

        return {
            "wave": wave,
            "period": period,
            "scale": scale,
            "power": power,
            "coi": coi,
            "avg_power": avg_power,
        }

    # ----------------------------
    # Significance (confidence interval)
    # ----------------------------
    def confidence_interval(
        self,
        wlt: Dict[str, np.ndarray],
        dat: ArrayLike,
        conf: float,
        noise_type: str = "r",
    ) -> Dict[str, np.ndarray]:
        """
        Confidence interval for wavelet power spectrum (Torrence & Compo style).
        noise_type:
            - "r" red noise (AR(1) fitted)
            - "w" white noise
        """
        dat = self._to_1d_array(dat)
        na = len(dat)

        if noise_type == "r":
            model = sm.tsa.ARIMA(dat, order=(1, 0, 0), trend="n").fit()
            alpha = float(model.params[0])  # lag-1 autocorr
            print(f"Red Noise AR1 Coefficient: {alpha}")
        else:
            alpha = 0.0

        ps = np.asarray(wlt["period"])
        freq = 1.0 / ps
        lp = len(ps)
        ci = np.zeros(lp)

        dt    = float(self.config.dt)
        scale = np.asarray(wlt["scale"])
        
        for i in range(lp):
            fnorm = freq[i] * dt                                  # cycles per sample, dt-agnostic
            p   = (1 - alpha**2) / (1 + alpha**2 - 2*alpha*np.cos(2*np.pi*fnorm))
            dof = 2 * np.sqrt(1 + (na*dt / (2.32 * scale[i]))**2)  # scale, not period; keep dt
            ci[i] = p * (chi2.ppf(conf, dof) / dof) * np.var(dat)

        out = dict(wlt)
        out["sig"] = ci
        return out

    # ----------------------------
    # Reconstruction
    # ----------------------------
    def reconstruct(
        self,
        wt: Dict[str, np.ndarray],
        series_data: ArrayLike,
        lvl: float = 0.0,
        only_coi: bool = True,
        only_sig: bool = True,
        rescale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the series from wavelet coefficients.

        Returns:
            reconstructed (np.ndarray): reconstructed series
            sig_scales (np.ndarray): significant scales (rounded) that passed Avg_Power > sig
        """
        series_data = self._to_1d_array(series_data)
        dj = float(self.config.dj)
        dt = float(self.config.dt)

        wave = np.asarray(wt["wave"])
        power = np.asarray(wt["power"])
        avg_power = np.asarray(wt["avg_power"])
        scale = np.asarray(wt["scale"])

        sig = np.asarray(wt.get("sig")) if wt.get("sig") is not None else None

        nc = len(series_data)
        nr = len(scale)

        Period = wt["period"]
        coi = wt["coi"]

        rec_waves = np.zeros((nr, nc))
        
        for s_ind in range(nr):
            rec_waves[s_ind, :] = (
                (np.real(wave[s_ind, :]) / np.sqrt(scale[s_ind]))
                * dj
                * np.sqrt(dt)
                / (np.pi ** (-1 / 4) * 0.776)
            )

        # power threshold
        rec_waves = rec_waves * (power >= lvl)

        # significance mask
        sig_scales = np.array([])
        if only_sig and sig is not None:
            # globally significant by avg_power > sig
            sig_mask = avg_power > sig  # True = significant

            # which scales are *ever* inside the COI at any time?
            # Period has shape (nr,), coi has shape (nc,)
            within_coi_any = np.array([
                np.any(Period[s_ind] <= coi) for s_ind in range(nr)
            ])

            # final mask for printing: significant AND within COI at least once
            print_mask = sig_mask & within_coi_any

            if not print_mask.any():
                warnings.warn("No significant wavelet periods identified within COI.")

            sig_scales = np.around(scale[print_mask], decimals=2)

            if sig_scales.size > 0:
                print("Significant scales (within COI):")
                print(sig_scales)
            else:
                print("No significant scales identified within COI.")

            # keep reconstruction masking behavior the same: mask non-significant rows
            rec_waves[~sig_mask, :] = np.nan

        # cone of influence filter
        if only_coi:   
            for t in range(nc):
                for s_ind in range(nr):
                    if Period[s_ind] > coi[t]:
                        rec_waves[s_ind, t] = np.nan

        reconstructed = np.nansum(rec_waves, axis=0)

        if rescale and np.std(reconstructed) > 0:
            reconstructed = (reconstructed - np.mean(reconstructed)) * np.std(series_data) / np.std(reconstructed) + np.mean(series_data)

        return reconstructed, sig_scales

    # ----------------------------
    # Plotting
    # ----------------------------
    def plot_wavelet(
        self,
        plt_dataset: Dict[str, np.ndarray],
        siglvl: float,
        name: str = "",
        sigtest: str = "default",
    ) -> None:
        """
        Plot wavelet power spectrum + global wavelet spectrum.
        """
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 4, width_ratios=[0.1, 2, 1, 0.1], wspace=0.1)

        start_color = "#FFFFFF"
        midpoint_color = "#6699CC"
        end_color = "#004488"
        custom_palette = sns.blend_palette([start_color, midpoint_color, end_color], as_cmap=True)

        ax_cwt = fig.add_subplot(gs[1])
        max_power = np.max(plt_dataset["Power"])
        levels = np.linspace(0, max_power, 10)

        time = plt_dataset["Time"]
        cs = ax_cwt.contourf(time, np.log2(plt_dataset["Period"]), plt_dataset["Power"], levels=levels, cmap=custom_palette)

        ax_cwt.plot(np.arange(time[0], time[0] + len(plt_dataset["COI"]), 1), np.log2(plt_dataset["COI"]), "k")
        ax_cwt.fill_between(
            np.arange(time[0], time[0] + len(plt_dataset["COI"]), 1),
            np.log2(plt_dataset["COI"]),
            np.log2(plt_dataset["Period"][-1]),
            color="gray",
            alpha=0.5,
        )
        ax_cwt.set_title("Local Wavelet Power Spectrum", fontsize=15)
        ax_cwt.set_ylabel("Period (Years)", fontsize=14)
        ax_cwt.set_xlabel("Time (Year)", fontsize=14)
        ax_cwt.tick_params(axis="both", which="major", labelsize=13)
        ax_cwt.set_ylim(np.log2(32), np.log2(2.1))
        ax_cwt.set_yticks(np.log2([4, 8, 16, 32]))
        ax_cwt.set_yticklabels([4, 8, 16, 32])

        cbar = plt.colorbar(cs, ax=ax_cwt, orientation="vertical")
        cbar.ax.tick_params(labelsize=13)
        cbar.formatter = FormatStrFormatter("%0.1f")
        cbar.update_ticks()

        ax_global = fig.add_subplot(gs[2])
        ax_global.plot(plt_dataset["Avg_Power"], plt_dataset["Period"], label="Power", linestyle="-")
        if sigtest == "red":
            ax_global.plot(plt_dataset["R_noise"], plt_dataset["Period"], color="red", label=f"Red Noise {int(siglvl*100)}%", linestyle="--")
        elif sigtest == "white":
            ax_global.plot(plt_dataset["W_noise"], plt_dataset["Period"], color="black", label=f"White Noise {int(siglvl*100)}%", linestyle="--")
        else:
            ax_global.plot(plt_dataset["R_noise"], plt_dataset["Period"], color="red", label=f"Red Noise {int(siglvl*100)}%", linestyle="--")
            ax_global.plot(plt_dataset["W_noise"], plt_dataset["Period"], color="black", label=f"White Noise {int(siglvl*100)}%", linestyle="--")

        ax_global.set_title("Global Wavelet Power Spectrum", fontsize=15)
        ax_global.set_xlabel("Power", fontsize=14)
        ax_global.tick_params(axis="both", which="major", labelsize=13)
        ax_global.set_yscale("log", base=2)
        ax_global.invert_yaxis()
        ax_global.set_xlim([0, 20])
        ax_global.set_ylim([32, 2.1])
        y_ticks = [2**x for x in range(2, 6)]
        ax_global.set_yticks(y_ticks)
        ax_global.set_yticklabels([str(y) for y in y_ticks])

        fig.suptitle(f"{name} Gauge", fontsize=17)
        plt.legend(fontsize=14)
        plt.show()

    # ----------------------------
    # Pipeline helper
    # ----------------------------
    def process_dataframe_columns(
            self,
            df: pd.DataFrame,
            cols: Sequence[str],
            siglvl: float,
            sigtest: str = "default",
            plot_wave: bool = False,
            drop_zero_frequency_months: bool = False,
            frequency_col_name: str = "Frequency",
        ) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        """
        Implements loop: for each column in `cols`,
        1) compute wavelet,
        2) compute CI for red and white noise,
        3) reconstruct using red-noise CI output,
        4) join reconstructed series as {col}_wave and lag as {col}_wave_lag,
        5) optionally drop rows where frequency == 0 (keeps NaNs as-is).
    
        Returns:
            (df_out, sig_scales_all)
        """
        df_out = df.copy()
        sig_scales_all: List[np.ndarray] = []
    
        # NEW: collect all new columns here instead of inserting one-by-one
        new_cols = {}
    
        for column in cols:
            print(f"Processing {column} ...")
    
            # copy + drop NaNs only for the wavelet computation
            col_series = df_out[column].copy()
            col_series = col_series.dropna()
    
            wlt = self.cwt(col_series.values)
            c_red = self.confidence_interval(wlt, col_series.values, siglvl, noise_type="r")
            c_white = self.confidence_interval(wlt, col_series.values, siglvl, noise_type="w")
    
            # time axis for plotting (monthly)
            year_month = col_series.index.to_period("M").to_timestamp()
            time = year_month.to_numpy()
    
            plt_dataset = {
                "Time": time,
                "Period": wlt["period"],
                "Avg_Power": wlt["avg_power"],
                "Power": wlt["power"],
                "COI": wlt["coi"],
                "W_noise": c_white["sig"],
                "R_noise": c_red["sig"],
            }
    
            if plot_wave:
                self.plot_wavelet(plt_dataset, siglvl, name=column, sigtest=sigtest)
    
            print("Red Noise Sig Test Results")
            reconstruction, sig_scales = self.reconstruct(c_red, col_series.values)
            sig_scales_all.append(sig_scales)
    
            # names
            wave_col_name = f"{column}_wave"
            wave_lag_col_name = f"{column}_wave_lag"
    
            # NEW: build full-length Series aligned to df_out.index
            wave_full = pd.Series(index=df_out.index, dtype=float)
            wave_full.loc[col_series.index] = reconstruction
    
            lag_full = wave_full.shift(1)
    
            # store into the dict instead of df_out[...] = ... inside loop
            new_cols[wave_col_name] = wave_full.values
            new_cols[wave_lag_col_name] = lag_full.values
    
        # NEW: add all new wave / lag columns in one shot -> no fragmentation
        if new_cols:
            df_out = pd.concat(
                [df_out, pd.DataFrame(new_cols, index=df_out.index)],
                axis=1
            )
    
        if drop_zero_frequency_months and frequency_col_name in df_out.columns:
            # only drop rows where frequency is exactly 0 (common in event-count series)
            df_out = df_out.loc[~(df_out[frequency_col_name] == 0)]
    
        return df_out, sig_scales_all

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _to_1d_array(x: ArrayLike) -> np.ndarray:
        if isinstance(x, pd.Series):
            arr = x.values
        else:
            arr = np.asarray(x)
        if arr.ndim != 1:
            arr = np.ravel(arr)
        return arr.astype(float)
