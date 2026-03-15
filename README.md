# Remifentanil Dose–Response and Haemodynamic Rebound: A VitalDB Cohort Study

**Analysis code** for the manuscript:

> *Dose-Dependent Inverse Association Between Intraoperative Remifentanil Exposure and Post-Cessation Haemodynamic Rebound: A Retrospective Cohort Study of 5,765 Surgical Patients*

Submitted to the *British Journal of Anaesthesia (BJA)*.

---

## Overview

This repository contains the minimum reproducible code for all analyses reported in the manuscript and supplementary materials. Raw data are not included because they are sourced from the [VitalDB open dataset](https://vitaldb.net/) and can be freely downloaded using the provided extraction scripts.

## Repository Structure

```
scripts/
├── oih_01_data_extraction.py      # VitalDB case screening & eligibility
├── oih_01b_batch_download.py      # Async batch download of vital signs
├── oih_01c_fast_download.py       # Optimised parallel download pipeline
├── oih_02_statistical_analysis.py # Primary analysis: RCS, correlations, dose–response
├── oih_02b_extended_analysis.py   # Extended analysis: alternative metrics, subgroups
├── oih_03_visualization.py        # Core figure generation
├── oih_04_reviewer_analyses.py    # Reviewer-requested analyses (Round 1)
├── oih_05_volatile_download.py    # Volatile anaesthetic data extraction
├── oih_06_reviewer2_analyses.py   # Reviewer-requested analyses (Round 2)
├── oih_07_sensitivity_supplement.py # Sensitivity & supplement analyses
├── oih_08_extended_analyses.py    # GPS, IPTW, segmented regression, E-max

figure_generation/
├── generate_v18_figures.py        # Main text Figures 2–4 (composite panels)
├── generate_main_tables.py        # Table 1 (baseline) & Table 2 (results)
├── generate_etables.py            # Supplementary tables (eTables S1–S18)
├── generate_strobe_flow.py        # Figure 1 (STROBE flow diagram)
├── regenerate_main_figures.py     # Legacy figure generation (v17)
```

## Data Source

All data are from the **VitalDB** open-access vital signs database:
- Website: https://vitaldb.net/
- API: https://api.vitaldb.net/
- Reference: Lee HC, et al. *Sci Data.* 2022;9:412.

To reproduce the dataset, run scripts in order:
```bash
python scripts/oih_01_data_extraction.py    # Screen eligible cases
python scripts/oih_01b_batch_download.py    # Download vital signs
python scripts/oih_02_statistical_analysis.py  # Run primary analysis
```

## Key Analyses

| Analysis | Script | Method |
|----------|--------|--------|
| Primary dose–response | `oih_02_statistical_analysis.py` | Restricted cubic splines (RCS) |
| Alternative exposure metrics | `oih_02b_extended_analysis.py` | Mean rate, AUC-Ce, peak Ce |
| Generalised propensity score | `oih_08_extended_analyses.py` | Hirano–Imbens GPS |
| IPTW causal inference | `oih_08_extended_analyses.py` | Stabilised IPTW |
| Taper dynamics | `oih_02b_extended_analysis.py` | Partial correlation, residualisation |
| Baseline sensitivity | `oih_07_sensitivity_supplement.py` | 10/15/30-min windows |
| E-max pharmacological model | `oih_08_extended_analyses.py` | 3-parameter E-max with bootstrap |
| Segmented regression | `oih_08_extended_analyses.py` | Piecewise linear breakpoint |

## Requirements

```
python >= 3.9
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
statsmodels
aiohttp          # for async VitalDB downloads
```

## License

This code is provided for academic reproducibility. The VitalDB data are governed by the [VitalDB Data Use Agreement](https://vitaldb.net/).

## Contact

For questions about the analysis, please contact the corresponding author (see manuscript).
