# Notebooks — exploratory analysis & interpretation

This folder contains **optional Jupyter notebooks** that support exploration,
interpretation, and communication of the results produced by `sar101.py`.

They are **not required** to run the pipeline.

The authoritative, repeatable workflow lives in `src/sar101.py`.
The notebooks exist to help you **understand, tune, and explain** the results.

---

## Intended workflow

The recommended workflow is:

```
1. Run the pipeline (sar101.py)
        ↓
2. Explore results interactively in notebooks
        ↓
3. Tune thresholds and parameters
        ↓
4. Re-run the pipeline with improved settings
```

This mirrors how SAR analytics are typically developed in practice:
**prototype → inspect → adjust → operationalize**.

---

## Step 1 — Run the pipeline

From the repository root:

```bash
python src/sar101.py
```

This produces raster and vector outputs in `outputs/` by default.

To compare polarizations (recommended):

```bash
python src/sar101.py --prefer-polarization vv --outdir outputs_vv
python src/sar101.py --prefer-polarization vh --outdir outputs_vh
```

---

## Step 2 — Explore in notebooks

Launch Jupyter:

```bash
jupyter lab
```

Open notebooks in this order:

### 🔬 01_explore_backscatter.ipynb
Purpose:
- Inspect **ratio_db = db(t1) - db(t0)**
- Plot histograms
- Compare **VV vs VH**
- Visually choose a good `--change-thr-db`

Typical questions answered:
- What does “normal” change look like?
- Where do the tails of the distribution start?
- How noisy is VH compared to VV?

---

### 🌊 02_water_vs_land.ipynb
Purpose:
- Inspect the **water-like mask**
- Reason about **false positives**
- Understand **speckle** and surface effects
- Decide whether `--water-thr-db` is too strict or too loose

This notebook intentionally emphasizes **visual reasoning**.
For detailed spatial inspection, QGIS is often the best tool.

---

### 📊 03_change_detection_explained.ipynb
Purpose:
- Explain the **physical meaning** of:
  - `db(t0)`
  - `db(t1)`
  - `ratio_db`
- Understand what **positive vs negative change** means
- Quantify how much area is flagged at different thresholds

This notebook is especially useful for:
- explaining results to non-SAR experts
- interview walkthroughs
- documentation and reporting

---

## Step 3 — Tune parameters

Based on what you see in the notebooks, adjust parameters such as:

- `--water-thr-db`
- `--change-thr-db`
- `--prefer-polarization`
- `--days`
- `--window-size`

Example:

```bash
python src/sar101.py \
  --water-thr-db -22 \
  --change-thr-db 1.5 \
  --prefer-polarization vv
```

---

## Step 4 — Re-run and validate

Re-run the pipeline with updated parameters and:
- re-open notebooks
- re-check outputs in QGIS
- confirm that false positives / noise are reduced

Repeat until results are stable and interpretable.

---

## Philosophy

- **Scripts** (`src/`) are for repeatability and deployment
- **Notebooks** are for thinking, tuning, and explaining

Keeping these roles separate makes the system easier to reason about,
debug, and communicate.
