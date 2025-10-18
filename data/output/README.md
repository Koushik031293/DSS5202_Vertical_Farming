# VF_LCIA_Auto — Stage-by-Stage LCIA for Vertical Farming

This script builds a complete LCIA workbook (per kg & per year), amortizes Capex, and outputs **stage-by-stage** impacts across six categories: GWP100, HOFP, PMFP, AP, EOFP, FFP.

## Input
- One sheet with `Parameter|Value` (annual output, kWh/kg, water, fertilizer/pesticide per ha + productivity, etc.).
- A sheet named **Capex** (either direct impacts per asset, mass×EF proxy, or cost×EF proxy).
- (Optional) A sheet **EF_Defaults** to override default factors.
- (Optional) A sheet **Normalization**: `category | norm_value`.

## Run
```bash
python vf_lcia_auto_capex.py \
  -i VF_LCC_Template_Singapore_FIXED.xlsx \
  -o VF_LCIA_ready_multiimpact.xlsx \
  --prefer-split-electricity --prefer-water-m3 \
  --write-readme
```

## plots
```bash
python src/vf_lcia_charts.py \
  -i data/output/VF_LCIA_ready_multiimpact.xlsx \
  -o data/output/charts \
  --fmt png --dpi 160 \
  --legend-pie --dominance-threshold 0.9 --min-slice-pct 0.02 \
  --per-cat --logy-per-cat \
  --normalized-by-stage
```

## Output sheets
Inputs_*, Derived_perkg, **LCI_perkg**, Annual_inventory, EF_Defaults,
**LCIA_rows_multi**, **LCIA_by_stage**, **LCIA_totals_multi**,
Normalized_by_stage (opt), Normalized_totals (opt).
