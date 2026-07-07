# Attention-Based Stochastic Simulation of Climate-Informed Spatiotemporal Flood Risk

Code accompanying:

> Nayak, A., Gentine, P., and Lall, U. *Attention-Based Stochastic Simulation of Climate-Informed
> Spatiotemporal Flood Risk for Multisite Insurance Portfolios.*
> *Geophysical Research Letters* (in press).
> DOI: `<add on publication>`

---

## Overview

This repository implements a climate-conditioned stochastic simulator for
spatiotemporal flood risk across a network of stream gauges, designed to
generate large synthetic ensembles for multisite insurance-portfolio loss
analysis. The pipeline couples a wavelet decomposition of large-scale climate
signals to an attention-based sequence model, retrieves climate analogs in the
learned embedding space, and reconstructs joint flood characteristics
(frequency, intensity, duration) with preserved cross-site and temporal
dependence.

The end-to-end pipeline is:

1. **Climate signal extraction** — Morlet continuous wavelet transform of
   large-scale climate indices (`wavelet_class.py`).
2. **Attention-based forecasting** — a transformer/Autoformer encoder producing
   point forecasts, optionally blended with a k-nearest-neighbour (kNN)
   datastore, with kNN residual resampling for ensemble spread
   (`transformer_knn.py`).
3. **Analog retrieval** — nearest historic climate windows are drawn in the
   encoder embedding space, either jointly across sites (`joint_analog.py`) or
   via analog-quantile matching against fitted marginals
   (`analog_quantile_sampler.py`).
4. **Marginal + dependence modelling** — parametric marginals for flood
   characteristics (`marginals_pos.py`) and a rank-reordering empirical copula
   (`empirical_copula.py`).
5. **Ensemble propagation** — large synthetic ensembles feed downstream
   insurance/solvency analysis.

The primary case study is a 117-gauge network in the Mississippi River Basin.

---

## Repository structure

### Core modules (`.py`)

| File | Role | Key entry points |
|------|------|------------------|
| `preprocess_data.py` | Data loading, cleaning, event extraction, train/val/test splitting; NWIS daily-discharge retrieval | `HydroClimateDataset`, `CVSplitConfig`, `extract_gauge_events`, `events_to_export` |
| `gauge_check.py` | Discover and filter NWIS daily-value discharge gauges in Mississippi-basin HUC2s | `MississippiGaugeChecker` |
| `trajectories.py` | Fetch NWIS instantaneous-value (sub-daily) records; refit sub-daily hydrograph templates | `fetch_iv_window`, `SubdailyHydrographRefitter` |
| `wavelet_class.py` | Morlet CWT of climate signals with significance testing and reconstruction | `WaveletAnalyzer`, `WaveletConfig` |
| `transformer_knn.py` | Attention-based forecaster: point forecast, kNN mean blending, kNN residual resampling | `HybridKNNTransformer`, `knn_topk`, `create_sequences_np` |
| `joint_analog.py` | Joint analog simulation ("Approach D") on the encoder embedding | `JointAnalogSampler`, `JointAnalogConfig` |
| `marginals_pos.py` | Univariate marginals (Poisson, NegBin, Exponential, Gamma, LogNormal, Weibull, GEV, truncated + log-spline) with auto shift-to-positive | `make_marginal`, `fit_with_auto_shift` |
| `empirical_copula.py` | Rank-reordering empirical copula simulator (Lall et al., 2016) + log-spline marginal surrogate | `EmpiricalCopulaSimulator`, `LogSplineMarginal` |
| `analog_quantile_sampler.py` | Analog-quantile matched copula sampler tying climate state to marginal level | `AnalogQuantileSampler` |
| `KNN_NS.py` | Diagnostics and utilities: QQ plots, tail-dependence, kNN weighting kernels | `plot_qq_multisite`, `tail_corr_matrix`, `KNNConditionalCopulaSampler` |
| `baselines.py` | Three comparison generators + shared CRPS/QQ evaluation | `LallSharmaKNN`, `SeasonalAR`, `NeymanScottBaseline` |

### Notebooks

| Notebook | Purpose | Paper artifact |
|----------|---------|----------------|
| `Main.ipynb` | End-to-end driver: preprocessing → transformer → analog/copula → ensemble | main results |
| `Wrapper.ipynb` | Orchestration / cross-validation runs (papermill) | — |

---

## Data sources

- **Streamflow** — USGS NWIS daily-value (`00060`, mean daily discharge) and
  instantaneous-value records, retrieved programmatically via
  `dataretrieval`/NWIS web services (`gauge_check.py`, `trajectories.py`,
  `preprocess_data.py`).
- **Climate signals** — large-scale climate indices used to drive the wavelet
  decomposition (`wavelet_class.py`).

No raw data are redistributed in this repository; all gauge data are pulled
from public USGS services at run time.

---

## Reproducibility notes

- **Random seeds.** Stochastic components accept explicit seeds
  (`numpy.random.default_rng`, `torch.manual_seed`, `random_state=`). Set these
  in the driver notebook for bit-for-bit reproducibility of a given ensemble.
- **Live gauge retrieval.** The gauge-selection and data-fetch steps query live
  USGS NWIS. The returned set can shift over time as stations are added,
  revised, or retired. To freeze the exact 117-gauge case study, cache the
  resolved gauge list (and downloaded series) to disk and load from the cache
  rather than re-querying. *Recommended: commit the resolved gauge list as a
  static CSV alongside the code.*

---

## Baselines

Three stochastic flood generators are provided for comparison, all fit on
training data only and evaluated through a shared CRPS/QQ interface
(`baselines.py`):

1. **`LallSharmaKNN`** — Lall & Sharma (1996) kNN bootstrap.
2. **`SeasonalAR`** — periodic AR(2) on monthly-max anomalies.
3. **`NeymanScottBaseline`** — independent parametric sampling per month
   (Poisson frequency, Exponential intensity and duration).

---

## Citation

Please cite the GRL article above if you use this code. A `CITATION.cff` can be
added once the DOI is assigned.

## License

See [`LICENSE`](LICENSE).
