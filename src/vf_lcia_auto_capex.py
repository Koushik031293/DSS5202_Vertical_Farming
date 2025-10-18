#!/usr/bin/env python3
# vf_lcia_auto_capex.py
# Build stage-by-stage LCIA (multi-impact) for a vertical farm:
# - Parses a Parameters sheet (Parameter|Value) to resolve per-kg flows
# - Amortizes Capex (LEDs, racks, HVAC, building...) from a 'Capex' sheet
# - Merges or creates EF_Defaults with 6 impact categories
# - Computes LCIA per flow, by stage, totals per kg & per year
# - Optional normalization (sheet 'Normalization': category|norm_value)
# - Optional README.md output (--write-readme / --readme-path)

import argparse, os
import pandas as pd
import numpy as np

# ------------------------ README (optional writer uses this) ------------------------
README_TEXT = r"""# VF_LCIA_Auto — Stage-by-Stage LCIA for Vertical Farming

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

## Output sheets
Inputs_*, Derived_perkg, **LCI_perkg**, Annual_inventory, EF_Defaults,
**LCIA_rows_multi**, **LCIA_by_stage**, **LCIA_totals_multi**,
Normalized_by_stage (opt), Normalized_totals (opt).
"""

# ------------------------ CORE SETTINGS ------------------------
IMPACT_COLS = ["GWP100","HOFP","PMFP","AP","EOFP","FFP"]
UNIT_MAP = {"GWP100":"kgCO2e","HOFP":"kgNMVOCeq","PMFP":"kgPM2.5eq","AP":"kgSO2eq","EOFP":"kgPO4eq","FFP":"MJ"}

# Default EF table (replace with ReCiPe/CML when available)
EF_DEFAULTS = pd.DataFrame([
    ["Electricity, medium voltage","kWh", 0.408, 2.8e-4, 1.8e-4, 2.8e-4, 1.0e-4, 7.6, "EMA/ReCiPe approx"],
    ["Water, tap","m3",            0.344, 1.0e-4, 5.0e-5, 1.0e-4, 1.0e-5, 2.5, "PUB/CML approx"],
    ["Sewerage, treatment","m3",   0.708, 1.0e-4, 8.0e-5, 2.0e-4, 1.5e-4, 1.5, "WWTP avg approx"],
    ["Fertilizer, NPK (as applied)","kg",4.000, 1.0e-3, 7.0e-4, 1.5e-2, 3.0e-3, 60.0,"CML/Agribalyse approx"],
    ["Pesticide, active ingredient","kg",25.000, 2.0e-3, 6.0e-3, 3.0e-2, 1.0e-2,100.0,"CML/FAO approx"],
    ["Packaging, plastic (generic)","kg", 2.700, 1.0e-3, 7.0e-4, 4.0e-3, 1.0e-3, 70.0,"PlasticsEurope approx"],
    ["Distribution, refrigerated van-km","tkm",0.180, 6.0e-4, 5.0e-4, 9.0e-4, 1.0e-4, 3.0,"Ecoinvent/CML approx"],
], columns=["flow_name","unit"] + IMPACT_COLS + ["source"])

# ------------------------ HELPERS ------------------------
def _num(x, default=None):
    try:
        return float(str(x).replace(",","").strip())
    except:
        return default

def lookup(params_df, names):
    for nm in names:
        row = params_df.loc[params_df["Parameter"].astype(str).str.strip() == nm]
        if not row.empty:
            return _num(row["Value"].iloc[0], None)
    return None

def build_perkg_from_inputs(params_df, prefer_split_electricity=False, prefer_water_m3=False):
    ao = lookup(params_df, ["Annual edible output (kg/yr)"])
    if not ao or ao <= 0:
        raise RuntimeError("Annual edible output (kg/yr) missing/invalid.")

    # Electricity (kWh/kg)
    elec = None
    if prefer_split_electricity:
        light = lookup(params_df, ["Lighting kWh per kg"])
        other = lookup(params_df, ["Other kWh per kg (HVAC, pumps, etc.)"])
        if light is not None and other is not None:
            elec = light + other
    if elec is None:
        ei = lookup(params_df, ["Electricity Intensity","Electricity Intensity (kWh/kg)"])
        if ei is not None: elec = ei
    if elec is None:
        total = lookup(params_df, ["Total annual energy","Electricity Consumption"])
        if total is not None: elec = total / ao

    # Water (m3/kg)
    water = None
    if prefer_water_m3:
        water = lookup(params_df, ["Water m3 per kg","Water (m3/kg)"])
    if water is None:
        wL = lookup(params_df, ["Water Intensity","Water Intensity (L/kg)"])
        if wL is not None: water = wL/1000.0
    if water is None:
        wA = lookup(params_df, ["Water Consumption"])
        if wA is not None: water = wA / ao

    sewer = lookup(params_df, ["Sewerage m3 per kg","Sewerage per kg (m3)"]) or water or 0.0

    # Fertilizer & pesticide from per-ha + productivity
    prod = lookup(params_df, ["Productivity (t/ha)","Productivity t/ha","Productivity"])
    fert_ha = lookup(params_df, ["Fertilizer Use","Fertilizer Use (kg/ha)"])
    pest_ha = lookup(params_df, ["Pesticide Use","Pesticide (kg a.i./ha)"])
    fert = (fert_ha/(prod*1000.0)) if (fert_ha and prod) else 0.0
    pest = (pest_ha/(prod*1000.0)) if (pest_ha and prod) else 0.0

    # Optional placeholders
    pack = lookup(params_df, ["Packaging mass per kg (kg)"]) or 0.0
    tkm  = lookup(params_df, ["Refrigerated tkm per kg"]) or 0.0

    return {
        "Annual output (kg/yr)": ao,
        "Electricity (kWh/kg)": elec or 0.0,
        "Water (m3/kg)": water or 0.0,
        "Sewer (m3/kg)": sewer or 0.0,
        "Fertilizer (kg/kg)": fert,
        "Pesticide (kg/kg)": pest,
        "Packaging (kg/kg)": pack,
        "Refrig tkm/kg": tkm
    }

def build_lci_perkg(d):
    """Return LCI with a 'stage' column for stage-by-stage LCIA."""
    rows = []
    def add(stage, name, cat, unit, val, note=""):
        rows.append({
            "stage": stage, "flow_name": name, "category": cat,
            "unit": unit, "amount_per_kg": float(val or 0.0), "note": note
        })

    add("Electricity",   "Electricity, medium voltage", "energy",      "kWh", d["Electricity (kWh/kg)"], "SG grid")
    add("Water",         "Water, tap",                  "water",       "m3",  d["Water (m3/kg)"],        "PUB water")
    add("Sewerage",      "Sewerage, treatment",         "water",       "m3",  d["Sewer (m3/kg)"],        "WWTP")
    add("Fertilizer",    "Fertilizer, NPK (as applied)","agrochemicals","kg",  d["Fertilizer (kg/kg)"],   "NPK")
    add("Pesticide",     "Pesticide, active ingredient","agrochemicals","kg",  d["Pesticide (kg/kg)"],    "a.i.")
    add("Packaging",     "Packaging, plastic (generic)","materials",   "kg",  d["Packaging (kg/kg)"],    "PET/PP/PE")
    add("Transport",     "Distribution, refrigerated van-km","transport","tkm",d["Refrig tkm/kg"],       "gate+ optional")
    return pd.DataFrame(rows)

def read_capex_amortization(xls, annual_output_kg):
    """Read 'Capex' and return per-kg proxy impact rows for capital goods (stage='Capital')."""
    if "Capex" not in xls.sheet_names:
        return pd.DataFrame(columns=["stage","flow_name","category","unit","amount_per_kg","note"])

    cap = pd.read_excel(xls, sheet_name="Capex")
    cap.columns = [str(c).strip() for c in cap.columns]
    for c in ["lifetime_years","mass_kg","ef_GWP100_kgCO2e_perkg","capex_SGD","ef_GWP100_kgCO2e_perSGD"] + IMPACT_COLS:
        if c in cap.columns:
            cap[c] = pd.to_numeric(cap[c], errors="coerce")

    if "asset" not in cap.columns or "lifetime_years" not in cap.columns:
        return pd.DataFrame(columns=["stage","flow_name","category","unit","amount_per_kg","note"])

    rows = []
    for _, r in cap.iterrows():
        asset = str(r.get("asset","Capital item"))
        life  = float(r.get("lifetime_years", 0) or 0)
        if life <= 0:
            continue

        # A) direct totals per category
        direct = False
        for cat in IMPACT_COLS:
            if cat in cap.columns and pd.notna(r.get(cat)):
                total = float(r[cat])
                perkg = total / (annual_output_kg * life)
                rows.append(["Capital", f"{asset} — {cat}", "materials", UNIT_MAP[cat], perkg, f"{cat} amortized per kg"])
                direct = True
        if direct:
            continue

        # B) mass × EF for GWP
        mass = float(r.get("mass_kg", 0) or 0)
        efkg = float(r.get("ef_GWP100_kgCO2e_perkg", 0) or 0)
        if mass > 0 and efkg > 0:
            total = mass * efkg
            perkg = total / (annual_output_kg * life)
            rows.append(["Capital", f"{asset} — GWP100", "materials", "kgCO2e", perkg, "mass×EF amortized per kg"])
            continue

        # C) cost × EF for GWP
        capex = float(r.get("capex_SGD", 0) or 0)
        ef_per_sgd = float(r.get("ef_GWP100_kgCO2e_perSGD", 0) or 0)
        if capex > 0 and ef_per_sgd > 0:
            total = capex * ef_per_sgd
            perkg = total / (annual_output_kg * life)
            rows.append(["Capital", f"{asset} — GWP100", "materials", "kgCO2e", perkg, "Cost×EF amortized per kg"])

    return pd.DataFrame(rows, columns=["stage","flow_name","category","unit","amount_per_kg","note"])

def merge_or_create_ef_defaults(xls, ef_defaults_df):
    if "EF_Defaults" in xls.sheet_names:
        cur = pd.read_excel(xls, sheet_name="EF_Defaults")
        for c in ["flow_name","unit"] + IMPACT_COLS:
            if c not in cur.columns:
                cur[c] = np.nan
        merged = pd.merge(ef_defaults_df, cur, on=["flow_name","unit"], how="outer", suffixes=("_new",""))
        for cat in IMPACT_COLS + ["source"]:
            col_new = f"{cat}_new" if f"{cat}_new" in merged.columns else None
            if col_new:
                merged[cat] = merged[cat].where(merged[cat].notna(), merged[col_new])
                merged = merged.drop(columns=[col_new])
        return merged
    return ef_defaults_df.copy()

def characterize_multi(lci_base, ef_df, annual_output_kg):
    """Characterize per flow; return row-level impacts + totals + stage pivot."""
    # split proxies (already in impact units) from regular flows
    is_proxy = lci_base["unit"].isin(UNIT_MAP.values())
    lci = lci_base[~is_proxy].copy()
    proxy = lci_base[is_proxy].copy()

    # merge EF for regular flows
    ef_use = ef_df.copy()
    id_cols = ["flow_name","unit"]
    for c in IMPACT_COLS:
        if c not in ef_use.columns:
            ef_use[c] = 0.0
    joined = lci.merge(ef_use[id_cols+IMPACT_COLS], on=id_cols, how="left")
    for c in IMPACT_COLS:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0.0)
        joined[f"{c}_perkg"] = joined["amount_per_kg"] * joined[c]

    # add proxy rows
    if not proxy.empty:
        for c in IMPACT_COLS:
            proxy[c+"_perkg"] = 0.0
        inv_unit_to_cat = {v:k for k,v in UNIT_MAP.items()}
        for idx, r in proxy.iterrows():
            cat = inv_unit_to_cat.get(str(r["unit"]))
            if cat:
                proxy.at[idx, cat+"_perkg"] = float(r["amount_per_kg"] or 0.0)
        joined = pd.concat([joined, proxy], ignore_index=True, sort=False)

    # totals per kg & per year
    totals = pd.DataFrame({
        "category": IMPACT_COLS,
        "perkg_total": [joined[f"{c}_perkg"].sum() for c in IMPACT_COLS]
    })
    totals["per_year_total"] = totals["perkg_total"] * annual_output_kg

    # stage-by-stage per kg
    by_stage = []
    for c in IMPACT_COLS:
        s = joined.groupby("stage", as_index=False)[f"{c}_perkg"].sum().rename(columns={f"{c}_perkg": c})
        by_stage.append(s.set_index("stage"))
    by_stage_df = pd.concat(by_stage, axis=1).reset_index().fillna(0.0)

    return joined, totals, by_stage_df

def maybe_normalize(totals_df, by_stage_df, norm_df):
    """Optional normalization using sheet 'Normalization' with columns: category | norm_value."""
    if norm_df is None or norm_df.empty:
        return None, None
    n = norm_df.copy()
    n.columns = [c.strip() for c in n.columns]
    if "category" not in n.columns or "norm_value" not in n.columns:
        return None, None
    # totals
    t = totals_df.merge(n, on="category", how="left")
    t["normalized_perkg"] = t["perkg_total"] / t["norm_value"]
    t["normalized_per_year"] = t["per_year_total"] / t["norm_value"]
    # by stage
    bs = by_stage_df.copy()
    for c in IMPACT_COLS:
        nv = n.loc[n["category"]==c, "norm_value"]
        if not nv.empty and pd.notna(nv.iloc[0]) and nv.iloc[0] != 0:
            bs[c+"_normalized"] = bs[c] / float(nv.iloc[0])
        else:
            bs[c+"_normalized"] = np.nan
    return t, bs

def write_readme(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(README_TEXT)

# ------------------------ MAIN ------------------------
def main():
    ap = argparse.ArgumentParser(description="Build stage-by-stage LCIA (multi-impact) with Capex amortization.")
    ap.add_argument("-i","--input", required=True, help="Path to Excel with Parameters + Capex.")
    ap.add_argument("-o","--output", default="VF_LCIA_ready_multiimpact.xlsx", help="Output workbook.")
    ap.add_argument("--prefer-split-electricity", action="store_true")
    ap.add_argument("--prefer-water-m3", action="store_true")
    ap.add_argument("--write-readme", action="store_true")
    ap.add_argument("--readme-path", default=None)
    args = ap.parse_args()

    xls = pd.ExcelFile(args.input)

    # find Parameter sheet
    params_df = None; pname=None
    for s in xls.sheet_names:
        df_try = pd.read_excel(xls, sheet_name=s)
        if any(str(c).strip().lower()=="parameter" for c in df_try.columns):
            params_df = df_try.copy(); pname=s; break
    if params_df is None:
        raise RuntimeError("No sheet with a 'Parameter' column found.")

    params_df.columns = [str(c).strip() for c in params_df.columns]
    params_df = params_df.dropna(subset=["Parameter"]).copy()

    # per-kg + LCI
    d = build_perkg_from_inputs(params_df, args.prefer_split_electricity, args.prefer_water_m3)
    annual_output_kg = d["Annual output (kg/yr)"]
    derived_tbl = pd.DataFrame(list(d.items()), columns=["Item","Value"])

    lci = build_lci_perkg(d)
    cap_perkg = read_capex_amortization(xls, annual_output_kg)
    if not cap_perkg.empty:
        lci = pd.concat([lci, cap_perkg], ignore_index=True)

    # Annual inventory
    annual = lci.copy()
    annual["annual_amount"] = annual["amount_per_kg"] * annual_output_kg

    # EF defaults (merge or create)
    ef = merge_or_create_ef_defaults(xls, EF_DEFAULTS)

    # Characterize + Stage pivot
    lcia_rows, lcia_totals, lcia_by_stage = characterize_multi(lci, ef, annual_output_kg)

    # Optional normalization
    norm_df = pd.read_excel(xls, sheet_name="Normalization") if "Normalization" in xls.sheet_names else None
    norm_totals, norm_by_stage = maybe_normalize(lcia_totals, lcia_by_stage, norm_df)

    # Write workbook
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as xw:
        params_df.to_excel(xw, sheet_name=f"Inputs_{(pname or 'Sheet1')[:28]}", index=False)
        derived_tbl.to_excel(xw, sheet_name="Derived_perkg", index=False)
        lci.to_excel(xw, sheet_name="LCI_perkg", index=False)
        annual.to_excel(xw, sheet_name="Annual_inventory", index=False)
        ef.to_excel(xw, sheet_name="EF_Defaults", index=False)
        lcia_rows.to_excel(xw, sheet_name="LCIA_rows_multi", index=False)
        lcia_by_stage.to_excel(xw, sheet_name="LCIA_by_stage", index=False)
        lcia_totals.to_excel(xw, sheet_name="LCIA_totals_multi", index=False)
        if norm_totals is not None:
            norm_by_stage.to_excel(xw, sheet_name="Normalized_by_stage", index=False)
            norm_totals.to_excel(xw, sheet_name="Normalized_totals", index=False)

    # README (optional)
    if args.write_readme:
        readme_path = args.readme_path or os.path.join(os.path.dirname(os.path.abspath(args.output)), "README.md")
        write_readme(readme_path)
        print(f"📝 Wrote README to: {readme_path}")

    print(f"✅ Wrote: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
