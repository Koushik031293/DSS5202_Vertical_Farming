# 🌱 Vertical vs Traditional Farming — LCIA Automation & Visualization

This repository automates **Life Cycle Impact Assessment (LCIA)** and **Life Cycle Costing (LCC)** analysis for **Singapore-based vertical and traditional farming systems**.  
It produces impact assessment workbooks and multi-format visualizations directly from structured Excel input data.

---

## 📂 Directory Structure

```
├── src/
│   ├── VF_TF_lcia_auto_capex.py          # Main LCIA automation script
│   ├── lcia_charts_unified.py            # Unified charting and reporting utility
│   ├── comprehensive_lcc_analysis.py     # Cost-based sustainability assessment
│   ├── complete_vf_analysis_singapore.py # Vertical farming LCIA workflow
│   ├── comprehensive_ahp_analysis.py     # AHP-based decision weighting
│   ├── npv_irr_sensitivity_analysis.py   # NPV/IRR scenario analysis
│   └── tf_lcia_auto_capex.py             # Traditional farming LCIA automation
│
├── data/
│   ├── input/
│   │   └── Corrected_Base_Data_Singapore.xlsx
│   └── output/
│       ├── VF_LCIA_ready_multiimpact.xlsx
│       └── TF_LCIA_ready_multiimpact.xlsx
│
├── charts_both/                          # Output visualizations (auto-created)
│   ├── VF_*                              # Charts for Vertical Farm
│   ├── TF_*                              # Charts for Traditional Farm
│
├── README.md                             # (You are here)
└── requirements.txt
```

---

## ⚙️ Environment Setup

Create and activate the project environment:

```bash
conda create -n lcia_env python=3.11 -y
conda activate lcia_env

pip install -r requirements.txt
```

Typical dependencies (already in requirements.txt):
```
pandas
numpy
matplotlib
openpyxl
argparse
pypandoc
```

---

## 🧮 Step 1: Generate LCIA Workbooks

Run the automation script to compute **LCIA totals**, **stage-wise impacts**, and **amortized CAPEX** for both systems.

### 🔹 Vertical Farming (VF)
```bash
python src/VF_TF_lcia_auto_capex.py   -i data/input/Corrected_Base_Data_Singapore.xlsx   -s VF   -o data/output/VF_LCIA_ready_multiimpact.xlsx   --write-readme
```

### 🔹 Traditional Farming (TF)
```bash
python src/VF_TF_lcia_auto_capex.py   -i data/input/Corrected_Base_Data_Singapore.xlsx   -s TF   -o data/output/TF_LCIA_ready_multiimpact.xlsx   --write-readme
```

✅ **Outputs generated:**
- Processed LCIA Excel files (`VF_LCIA_ready_multiimpact.xlsx`, `TF_LCIA_ready_multiimpact.xlsx`)
- Auto-generated README summary sheet within each Excel
- Intermediate amortization and category-level summaries

---

## 📊 Step 2: Generate LCIA Charts

Use the unified chart generator to visualize total and stage-wise environmental impacts across multiple categories.

### 🔹 Vertical Farm Charts
```bash
python src/lcia_charts_unified.py   -i data/output/VF_LCIA_ready_multiimpact.xlsx   -o charts_both   --label VF   --per-cat --normalized-by-stage   --write-md --md-file VF_report.md --md-title "Vertical Farm LCIA – Charts"   --images-only-pdf
```

### 🔹 Traditional Farm Charts
```bash
python src/lcia_charts_unified.py   -i data/output/TF_LCIA_ready_multiimpact.xlsx   -o charts_both   --label TF   --per-cat --normalized-by-stage   --write-md --md-file TF_report.md --md-title "Traditional Farm LCIA – Charts"   --images-only-pdf
```

✅ **Charts generated (examples):**
- `totals_bar.png` – Total impact per category (per kg)
- `by_stage_stacked_bar.png` – Stage-wise stacked comparison
- `normalized_radar.png` – Normalized multi-impact radar chart
- `normalized_by_stage_stacked.png` – Normalized per-stage (0–1 scale)
- `gwp_stage_pie.png` – GWP100 contribution by process stage
- `VF_report.md` / `TF_report.md` – Markdown summaries
- `VF_report.pdf` / `TF_report.pdf` – PDF of all charts (image-only format)

---

## 📈 Step 3: Advanced Analysis (Optional)

Additional modules available:

| Script | Description |
|--------|--------------|
| `comprehensive_lcc_analysis.py` | Computes cost breakdowns and compares per-kg cost structures |
| `complete_vf_analysis_singapore.py` | Full LCIA pipeline for vertical farms |
| `comprehensive_ahp_analysis.py` | AHP-based ranking of sustainability priorities |
| `npv_irr_sensitivity_analysis.py` | Economic sensitivity for CAPEX, NPV, and IRR under various scenarios |

Run any script using:
```bash
python src/<script_name>.py
```

---

## 📘 Generated Reports

| File | Description |
|------|--------------|
| `VF_LCIA_ready_multiimpact.xlsx` | LCIA workbook (VF) |
| `TF_LCIA_ready_multiimpact.xlsx` | LCIA workbook (TF) |
| `VF_report.md / VF_report.pdf` | Charts and summary for Vertical Farming |
| `TF_report.md / TF_report.pdf` | Charts and summary for Traditional Farming |
| `charts_both/` | Folder containing all generated images |

---

## 📑 Notes

- Ensure the **input Excel file** contains all necessary sheets (`Inventory`, `Capex`, `Impact Factors`, etc.).
- If `--write-readme` is passed, a README sheet summarizing assumptions and parameters will be added to the workbook.
- All charts and reports are reproducible from the saved Excel outputs.

---

