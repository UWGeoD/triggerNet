# triggerNet

A Python pipeline for clustering energy releases (e.g., earthquakes, microseismicity, or acoustic emissions) catalogs using nearest-neighbor distance (NND) in the space-time-magnitude domain.  
Automatically detects event clusters, computes key statistics, and generates summary plots—supporting both research and exploratory data analysis.

![results_overlaid_volc_python](https://github.com/user-attachments/assets/68aa4619-3918-4830-ac76-09d74d65068a)


**Authors:** Ellie K. Johnson and Jesse C. Hampton

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Citation](#citation)
- [References](#references)

---

## Features

- **Flexible Input:** Works with CSV or MATLAB (.mat) catalogs.
- **Robust Parameterization:** Auto-estimates or uses user-provided b-value, fractal dimension, and other clustering parameters.
- **Cluster Extraction:** Builds a spanning tree and extracts cluster forests via thresholding.
- **Diagnostic Plots:** Generates histograms and contour plots comparing original vs. randomized catalogs.
- **Modular Design:** Core logic is cleanly separated for easy extension.
- **CLI Ready:** All paths and parameters are configurable via command line.

---

## Installation

Requires **Python 3.10+**

### 1. Clone the Repository

```bash
git clone https://github.com/UWGeoD/triggerNet.git
cd triggerNet
````

### 2. Install Dependencies

**Recommended:** Use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Or, with Conda:

```bash
conda env create -f environment.yml
conda activate triggerNet
```

---

## Quick Start

```bash
python src/main.py \
  -i data/catalog.csv \
  -o results \
  --mag_col magnitude \
  --x_col longitude \
  --y_col latitude
```

* Results and plots will appear in `/results` and `/plots` directories.

---

## Usage

**Command-Line Options:**

| Flag                    | Description                                             | Default    |
| ----------------------- | ------------------------------------------------------- | ---------- |
| `-i`, `--input`         | Path to input catalog (`.csv` or `.mat`)                | *required* |
| `-o`, `--output_prefix` | Prefix for output files                                 | `results`  |
| `--time_col`            | Name of time column in input                            | `time`     |
| `--x_col`               | Name of x (lon) column                                  | `x`        |
| `--y_col`               | Name of y (lat) column                                  | `y`        |
| `--z_col`               | Name of z (depth) column                                | None       |
| `--mag_col`             | Name of magnitude column                                | `mag`      |
| `--time_format`         | Datetime parsing format                                 | None       |
| `--mag_cutoff`          | Minimum magnitude to include                            | None       |
| `--b`                   | b-value (auto-estimated if not set)                     | None       |
| `--df`                  | Fractal dimension                                       | `1.6`      |
| `--eta0`                | Threshold for strong links (auto-estimated if not set)  | None       |

*Example:*

```bash
python src/main.py -i data/catalog.csv -o demo --mag_cutoff 2.0
```

**Outputs:**

* `results/<prefix>_nnd.csv`: Catalog with NND and parent assignments.
* `results/<prefix>_tree.csv`: Spanning tree edges.
* `results/<prefix>_adjacency.csv`: Strong-link adjacency matrix.
* `results/<prefix>_clusters.txt`: List of clusters (components).
* `plots/<prefix>_hist.png`: Histogram of log₁₀(η) (original and w/ shuffled).
* `plots/<prefix>_contour.png`: Contour plot (original and w/ shuffled).

---

## Repository Structure

```
triggerNet/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── data_io.py
│   ├── analysis.py
│   ├── clustering.py
│   └── utils.py
├── data/
├── plots/
├── results/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
```

* **src/**: Source code (main pipeline, analysis, clustering, utilities)
* **data/**: Input catalogs
* **plots/**: Output plots (autogenerated)
* **results/**: Output CSVs (autogenerated)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this pipeline, please cite as:

```
@software{johnson2025triggerNet,
  author    = {Johnson, Ellie K. and Hampton, Jesse C.},
  title     = {triggerNet: Nearest-Neighbor Energy Release Clustering Pipeline},
  year      = {2025},
  url       = {https://github.com/UWGeoD/triggerNet},
}
```

---

**Questions?**
Open an [issue](https://github.com/UWGeoD/triggerNet/issues) or email the maintainer (ekjohnson23@wisc.edu)!

---

## References

1. Aki, K. (1965). Maximum likelihood estimate of b in the formula $\log N = a - bM$ and its confidence limits. Bulletin of the Earthquake Research Institute, 43, 237–239.

2. Davidsen, J., Kwiatek, G., Charalampidou, E. M., Goebel, T., Stanchits, S., Rück, M., & Dresen, G. (2017). Triggering Processes in Rock Fracture. *Physical Review Letters, 119*(6). https://doi.org/10.1103/PhysRevLett.119.068501

3. Li, B. Q., Smith, J. D., & Ross, Z. E. (2021). Basal nucleation and the prevalence of ascending swarms in Long Valley caldera. *Science Advances, 7*(35), eabi8368. https://doi.org/10.1126/sciadv.abi8368

4. Zaliapin, I., & Ben-Zion, Y. (2013). Earthquake clusters in southern California I: Identification and stability. *Journal of Geophysical Research: Solid Earth, 118*(6), 2847–2864. https://doi.org/10.1002/jgrb.50179

