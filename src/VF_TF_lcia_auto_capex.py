#!/usr/bin/env python3
# VF_TF_lcia_auto_capex.py
#
# Stage-by-stage multi-impact LCIA for a chosen system (VF = vertical farm, TF = traditional farm).
# - Reads <SYSTEM>_Parameters and <SYSTEM>_CAPEX from a single workbook.
# - Computes per-kg inventory (electricity, water, etc.).
# - Amortizes Capex impacts over lifetime.
# - Joins with default or user-provided EF factors.
# - Produces per-stage LCIA, totals per kg and per year, optional normalization.

import argparse, os
import pandas as pd
import numpy as np

# ------------------------ CONSTANTS ------------------------

IMPACT_COLS = ["GWP100","HOFP","PMFP","AP","EOFP","FFP"]

UNIT_MAP = {
    "GWP100": "kgCO2e",
    "HOFP":   "kgNMVOCeq",
    "PMFP":   "kgPM2.5eq",
    "AP":     "kgSO2eq",
    "EOFP":   "kgPO4eq",
    "FFP":    "MJ",
}

README_TEXT = r"""Stage-by-stage LCIA with Capex amortization for VF/TF systems.

How to run:

python VF_TF_lcia_auto_capex.py \
  -i data/input/Corrected_Base_Data_Singapore.xlsx \
  -s VF \
  -o data/output/VF_LCIA_ready_multiimpact.xlsx \
  --write-readme
"""

# Default EF table. If your workbook has an EF_Defaults sheet, it will override these.
EF_DEFAULTS = pd.DataFrame([
    ["Electricity, medium voltage","kWh", 0.408, 2.8e-4, 1.8e-4, 2.8e-4, 1.0e-4, 7.6, "EMA/ReCiPe approx"],
    ["Water, tap","m3",            0.344, 1.0e-4, 5.0e-5, 1.0e-4, 1.0e-5, 2.5, "PUB/CML approx"],
    ["Sewerage, treatment","m3",   0.708, 1.0e-4, 8.0e-5, 2.0e-4, 1.5e-4, 1.5, "WWTP avg approx"],
    ["Fertilizer, NPK (as applied)","kg",4.000, 1.0e-3, 7.0e-4, 1.5e-2, 3.0e-3, 60.0,"CML/Agribalyse approx"],
    ["Pesticide, active ingredient","kg",25.000, 2.0e-3, 6.0e-3, 3.0e-2, 1.0e-2,100.0,"CML/FAO approx"],
    ["Packaging, plastic (generic)","kg", 2.700, 1.0e-3, 7.0e-4, 4.0e-3, 1.0e-3, 70.0,"PlasticsEurope approx"],
    ["Distribution, refrigerated van-km","tkm",0.180, 6.0e-4, 5.0e-4, 9.0e-4, 1.0e-4, 3.0,"Ecoinvent/CML approx"],
], columns=["flow_name","unit"] + IMPACT_COLS + ["source"])

# ------------------------ UTILS ------------------------

def _num(x, default=None):
    """Try to convert x to float. Return default if NaN or not parseable."""
    try:
        if pd.isna(x):
            return default
        return float(str(x).replace(",","").strip())
    except:
        return default

def _normcols(df):
    """Trim whitespace from column headers."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def lookup(params_df, names):
    """
    Find the first row in params_df where 'Parameter' exactly matches
    any string in `names`. Return its 'Value' as float.
    """
    for nm in names:
        row = params_df.loc[params_df["Parameter"].astype(str).str.strip() == nm]
        if not row.empty:
            return _num(row["Value"].iloc[0], None)
    return None

# ------------------------ PARAM SHEET READER ------------------------

def read_parameter_sheet_with_flexible_header(xls, sheet_name):
    """
    Many of these sheets start with a title row like:
    'VERTICAL FARM - CORRECTED PARAMETERS (Singapore)'
    and THEN come the actual headers 'Parameter | Value | Units | Notes'.

    We:
      1. read with header=None,
      2. scan first ~20 rows to find a row containing both 'Parameter' and 'Value',
      3. re-read using that row index as the header.
    """
    raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    header_row_idx = None
    for i in range(min(20, len(raw))):
        row_vals = [str(v).strip() for v in list(raw.iloc[i].values)]
        if "Parameter" in row_vals and "Value" in row_vals:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise RuntimeError(
            f"Could not find a header row with 'Parameter' and 'Value' in sheet '{sheet_name}'."
        )

    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row_idx)
    df = _normcols(df)

    if "Parameter" not in df.columns or "Value" not in df.columns:
        raise RuntimeError(f"Sheet '{sheet_name}' must have 'Parameter' and 'Value' columns.")

    df = df.dropna(subset=["Parameter"]).copy()
    return df

# ------------------------ PER-KG INTENSITIES ------------------------

def build_perkg_from_inputs(params_df):
    """
    Build per-kg intensities for flows based on VF_Parameters / TF_Parameters content.
    We rely on wording that matches your screenshot:
      - Annual edible output (kg/yr)
      - Lighting kWh per kg
      - HVAC & other kWh per kg
      - Water m3 per kg
      - Fertilizer intensity (kg/kg produce)
      - Pesticide intensity (kg/kg produce)
    We'll also guess sewer volume per kg = water per kg
    if not explicitly provided.
    """

    # Annual output
    annual_output = lookup(params_df, [
        "Annual edible output (kg/yr)",
        "Annual edible output",
        "Annual edible output (kg/yr) "
    ])
    if not annual_output or annual_output <= 0:
        raise RuntimeError("Annual edible output (kg/yr) missing/invalid.")

    # Electricity kWh/kg
    light_kwh = lookup(params_df, [
        "Lighting kWh per kg",
        "Lighting kWh per kg "
    ])
    other_kwh = lookup(params_df, [
        "HVAC & other kWh per kg",
        "HVAC & other kWh per kg "
    ])
    elec_kwh_perkg = 0.0
    if light_kwh is not None:
        elec_kwh_perkg += light_kwh
    if other_kwh is not None:
        elec_kwh_perkg += other_kwh
    # fallback if no split
    if elec_kwh_perkg == 0.0:
        fallback_e = lookup(params_df, [
            "Electricity Intensity (kWh/kg)",
            "Electricity Intensity"
        ])
        if fallback_e is not None:
            elec_kwh_perkg = fallback_e

    # Water m3/kg
    water_m3_perkg = lookup(params_df, [
        "Water m3 per kg",
        "Water m3 per kg ",
        "Water m3 per kg produce",
        "Water m3 per kg produce "
    ])
    if water_m3_perkg is None:
        # fallback via L/kg if ever provided
        wL = lookup(params_df, ["Water Intensity (L/kg)","Water Intensity"])
        if wL is not None:
            water_m3_perkg = wL / 1000.0
    if water_m3_perkg is None:
        water_m3_perkg = 0.0

    # Sewer m3/kg (not directly provided in screenshot, fallback to water)
    sewer_m3_perkg = lookup(params_df, [
        "Sewer m3 per kg",
        "Sewer (m3/kg)",
        "Sewerage m3 per kg"
    ])
    if sewer_m3_perkg is None or sewer_m3_perkg == 0.0:
        sewer_m3_perkg = water_m3_perkg

    # Fertilizer intensity kg/kg produce
    fert_perkg = lookup(params_df, [
        "Fertilizer intensity (kg/kg produce)",
        "Fertilizer intensity (kg/kg produce) "
    ]) or 0.0

    # Pesticide intensity kg/kg produce
    pest_perkg = lookup(params_df, [
        "Pesticide intensity (kg/kg produce)",
        "Pesticide intensity (kg/kg produce) "
    ]) or 0.0

    # Packaging mass per kg produce
    # If not provided, we default to 0. (Your sheet has cost, not mass.)
    pack_kg_perkg = lookup(params_df, [
        "Packaging mass per kg (kg)",
        "Packaging mass per kg",
        "Packaging (kg/kg)"
    ]) or 0.0

    # Cold transport tkm/kg
    # If you later add something like "Refrig tkm/kg", we'll pick it up.
    refrig_tkm_perkg = lookup(params_df, [
        "Refrigerated tkm per kg",
        "Refrig tkm/kg",
        "Transport tkm/kg"
    ]) or 0.0

    return {
        "Annual output (kg/yr)": annual_output,
        "Electricity (kWh/kg)": elec_kwh_perkg or 0.0,
        "Water (m3/kg)": water_m3_perkg or 0.0,
        "Sewer (m3/kg)": sewer_m3_perkg or 0.0,
        "Fertilizer (kg/kg)": fert_perkg,
        "Pesticide (kg/kg)": pest_perkg,
        "Packaging (kg/kg)": pack_kg_perkg,
        "Refrig tkm/kg": refrig_tkm_perkg,
    }

def build_lci_perkg(perkg_dict):
    """
    Build the life cycle inventory table per kg of product for operational flows.
    """
    rows = []
    def add(stage, name, cat, unit, val, note=""):
        rows.append({
            "stage": stage,
            "flow_name": name,
            "category": cat,
            "unit": unit,
            "amount_per_kg": float(val or 0.0),
            "note": note
        })

    add("Electricity", "Electricity, medium voltage", "energy", "kWh",
        perkg_dict["Electricity (kWh/kg)"], "grid electricity")
    add("Water", "Water, tap", "water", "m3",
        perkg_dict["Water (m3/kg)"], "supply water")
    add("Sewerage", "Sewerage, treatment", "water", "m3",
        perkg_dict["Sewer (m3/kg)"], "wastewater treatment")
    add("Fertilizer", "Fertilizer, NPK (as applied)", "agrochemicals", "kg",
        perkg_dict["Fertilizer (kg/kg)"], "on-crop fertilizer")
    add("Pesticide", "Pesticide, active ingredient", "agrochemicals", "kg",
        perkg_dict["Pesticide (kg/kg)"], "a.i. use")
    add("Packaging", "Packaging, plastic (generic)", "materials", "kg",
        perkg_dict["Packaging (kg/kg)"], "primary packaging")
    add("Transport", "Distribution, refrigerated van-km", "transport", "tkm",
        perkg_dict["Refrig tkm/kg"], "cold chain delivery")

    return pd.DataFrame(rows)

# ------------------------ CAPEX AMORTIZATION ------------------------

def read_capex_amortization(xls, capex_sheet, annual_output_kg):
    """
    Read <SYSTEM>_CAPEX and convert capital assets into per-kg burdens.
    Supports:
    - Direct impact columns (GWP100, HOFP, PMFP, AP, EOFP, FFP)
    - OR fallback GWP100-only via mass * EF or cost * EF.
    """
    cap = pd.read_excel(xls, sheet_name=capex_sheet)
    cap = _normcols(cap)

    # numeric cleanup
    for c in [
        "lifetime_years",
        "mass_kg",
        "ef_GWP100_kgCO2e_perkg",
        "capex_SGD",
        "ef_GWP100_kgCO2e_perSGD"
    ] + IMPACT_COLS:
        if c in cap.columns and c not in ["asset","stage","subsystem"]:
            cap[c] = pd.to_numeric(cap[c], errors="coerce")

    out_rows = []

    for _, r in cap.iterrows():
        asset = str(r.get("asset","Capital item")).strip()
        life  = _num(r.get("lifetime_years",0),0)
        if not life or life <= 0:
            # no lifetime => can't amortize
            continue

        # Stage/subsystem label for LCIA_by_stage
        stage_label = r.get("stage", None)
        if pd.isna(stage_label) or not stage_label:
            stage_label = r.get("subsystem", None)
        if pd.isna(stage_label) or not stage_label:
            stage_label = "Capital"

        # Case 1: direct multi-category data per asset
        has_any_full = any(
            (c in cap.columns and pd.notna(r.get(c)))
            for c in IMPACT_COLS
        )
        if has_any_full:
            for cat in IMPACT_COLS:
                val = r.get(cat, np.nan)
                if pd.isna(val):
                    continue
                perkg = float(val) / (annual_output_kg * life)
                out_rows.append({
                    "stage": stage_label,
                    "flow_name": f"{asset} — {cat}",
                    "category": "capital",
                    "unit": UNIT_MAP[cat],
                    "amount_per_kg": perkg,
                    "note": f"{cat} amortized per kg"
                })
            continue

        # Case 2: fallback climate-only via mass*EF
        mass = _num(r.get("mass_kg",0),0)
        efkg = _num(r.get("ef_GWP100_kgCO2e_perkg",0),0)
        if mass and efkg:
            total_gwp = mass * efkg
            perkg = total_gwp / (annual_output_kg * life)
            out_rows.append({
                "stage": stage_label,
                "flow_name": f"{asset} — GWP100",
                "category": "capital",
                "unit": UNIT_MAP["GWP100"],
                "amount_per_kg": perkg,
                "note": "mass×EF amortized"
            })
            continue

        # Case 3: fallback climate-only via cost*EF
        capex_val = _num(r.get("capex_SGD",0),0)
        ef_sgd    = _num(r.get("ef_GWP100_kgCO2e_perSGD",0),0)
        if capex_val and ef_sgd:
            total_gwp = capex_val * ef_sgd
            perkg = total_gwp / (annual_output_kg * life)
            out_rows.append({
                "stage": stage_label,
                "flow_name": f"{asset} — GWP100",
                "category": "capital",
                "unit": UNIT_MAP["GWP100"],
                "amount_per_kg": perkg,
                "note": "cost×EF amortized"
            })
            continue

    if not out_rows:
        return pd.DataFrame(columns=[
            "stage","flow_name","category","unit","amount_per_kg","note"
        ])

    return pd.DataFrame(out_rows, columns=[
        "stage","flow_name","category","unit","amount_per_kg","note"
    ])

# ------------------------ EF DEFAULTS MERGE ------------------------

def merge_or_create_ef_defaults(xls, ef_defaults_df):
    """
    If the workbook has an EF_Defaults sheet, merge it with the baseline EF_DEFAULTS.
    User-specified values override defaults where present.
    """
    if "EF_Defaults" in xls.sheet_names:
        cur = pd.read_excel(xls, sheet_name="EF_Defaults")
        cur = _normcols(cur)

        for c in ["flow_name","unit"] + IMPACT_COLS:
            if c not in cur.columns:
                cur[c] = np.nan

        merged = pd.merge(
            ef_defaults_df,
            cur,
            on=["flow_name","unit"],
            how="outer",
            suffixes=("_new","")
        )

        # prefer user override (no suffix), fallback to _new if user blank
        for cat in IMPACT_COLS + ["source"]:
            col_new = f"{cat}_new"
            if col_new in merged.columns:
                merged[cat] = merged[cat].where(
                    merged[cat].notna(), merged[col_new]
                )
                merged = merged.drop(columns=[col_new])

        return merged

    return ef_defaults_df.copy()

# ------------------------ CHARACTERIZATION ------------------------

def characterize_multi(lci_base, ef_df, annual_output_kg):
    """
    Calculate per-kg impacts for each flow, then:
    - totals per category,
    - by-stage breakdown.
    We treat capital amortization rows as already in final impact units.
    """

    # mark proxy rows = rows already expressed in impact units
    is_proxy = lci_base["unit"].isin(UNIT_MAP.values())
    lci_normal = lci_base[~is_proxy].copy()
    lci_proxy  = lci_base[is_proxy].copy()

    # 1. Join EF factors with normal rows
    ef_use = ef_df.copy()
    id_cols = ["flow_name","unit"]
    for c in IMPACT_COLS:
        if c not in ef_use.columns:
            ef_use[c] = 0.0

    joined = lci_normal.merge(
        ef_use[id_cols + IMPACT_COLS],
        on=id_cols,
        how="left"
    )

    for c in IMPACT_COLS:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0.0)
        joined[f"{c}_perkg"] = joined["amount_per_kg"] * joined[c]

    # 2. Add proxy rows: amount_per_kg is ALREADY in final units
    if not lci_proxy.empty:
        prox = lci_proxy.copy()
        # init all impact columns
        for c in IMPACT_COLS:
            prox[c+"_perkg"] = 0.0
        # map unit -> which category
        unit_to_cat = {v:k for k,v in UNIT_MAP.items()}
        for idx, r in prox.iterrows():
            cat = unit_to_cat.get(str(r["unit"]))
            if cat:
                prox.at[idx, cat+"_perkg"] = float(r["amount_per_kg"] or 0.0)
        joined = pd.concat([joined, prox], ignore_index=True, sort=False)

    # totals per kg and per year
    totals = pd.DataFrame({
        "category": IMPACT_COLS,
        "perkg_total": [joined[f"{c}_perkg"].sum() for c in IMPACT_COLS]
    })
    totals["per_year_total"] = totals["perkg_total"] * annual_output_kg

    # by-stage table
    by_stage_parts = []
    for c in IMPACT_COLS:
        st = joined.groupby("stage", as_index=False)[f"{c}_perkg"].sum().rename(
            columns={f"{c}_perkg": c}
        )
        by_stage_parts.append(st.set_index("stage"))
    by_stage_df = pd.concat(by_stage_parts, axis=1).reset_index().fillna(0.0)

    return joined, totals, by_stage_df

# ------------------------ NORMALIZATION (OPTIONAL) ------------------------

def maybe_normalize(totals_df, by_stage_df, norm_df):
    """
    Normalization sheet must have:
      category | norm_value
    We'll compute normalized_perkg and normalized_per_year, and add
    *_normalized cols to the stage table.
    """
    if norm_df is None or norm_df.empty:
        return None, None

    n = _normcols(norm_df)
    if "category" not in n.columns or "norm_value" not in n.columns:
        return None, None

    # totals normalization
    t = totals_df.merge(
        n[["category","norm_value"]],
        on="category",
        how="left"
    )
    t["normalized_perkg"] = t["perkg_total"] / t["norm_value"]
    t["normalized_per_year"] = t["per_year_total"] / t["norm_value"]

    # stage normalization
    bs = by_stage_df.copy()
    for c in IMPACT_COLS:
        nv = n.loc[n["category"]==c, "norm_value"]
        if not nv.empty and pd.notna(nv.iloc[0]) and nv.iloc[0] != 0:
            bs[c+"_normalized"] = bs[c] / float(nv.iloc[0])
        else:
            bs[c+"_normalized"] = np.nan

    return t, bs

# ------------------------ MAIN ------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Stage-by-stage LCIA with Capex amortization for VF or TF system."
    )
    ap.add_argument("-i","--input", required=True,
                    help="Excel workbook with *_Parameters, *_CAPEX, etc.")
    ap.add_argument("-s","--system", required=True, choices=["VF","TF"],
                    help="Which system to analyze: VF (vertical farm) or TF (traditional farm)")
    ap.add_argument("-o","--output", default=None,
                    help="Output workbook (.xlsx). Default is <system>_LCIA_ready_multiimpact.xlsx")
    ap.add_argument("--write-readme", action="store_true",
                    help="Also write a README.md next to output.")
    ap.add_argument("--readme-path", default=None,
                    help="Optional explicit README.md path.")
    args = ap.parse_args()

    system = args.system.strip().upper()  # "VF" or "TF"
    sheet_params = f"{system}_Parameters"
    sheet_capex  = f"{system}_CAPEX"

    if args.output is None:
        args.output = f"{system}_LCIA_ready_multiimpact.xlsx"

    # --- load workbook
    xls = pd.ExcelFile(args.input)

    # --- read parameter sheet
    if sheet_params not in xls.sheet_names:
        raise RuntimeError(f"Sheet '{sheet_params}' not found in workbook.")
    params_df = read_parameter_sheet_with_flexible_header(xls, sheet_params)

    # --- build per-kg intensities
    perkg_dict = build_perkg_from_inputs(params_df)
    annual_output_kg = perkg_dict["Annual output (kg/yr)"]
    derived_tbl = pd.DataFrame(list(perkg_dict.items()), columns=["Item","Value"])

    # --- LCI for operations
    lci_ops = build_lci_perkg(perkg_dict)

    # --- Capex amortization (if available)
    if sheet_capex in xls.sheet_names:
        cap_perkg = read_capex_amortization(xls, sheet_capex, annual_output_kg)
    else:
        cap_perkg = pd.DataFrame(columns=["stage","flow_name","category","unit","amount_per_kg","note"])

    # merge op + capex
    if not cap_perkg.empty:
        lci_all = pd.concat([lci_ops, cap_perkg], ignore_index=True)
    else:
        lci_all = lci_ops.copy()

    # --- build annual inventory
    annual_inventory = lci_all.copy()
    annual_inventory["annual_amount"] = annual_inventory["amount_per_kg"] * annual_output_kg

    # --- EF table (merge or override with EF_Defaults sheet if present)
    ef_table = merge_or_create_ef_defaults(xls, EF_DEFAULTS)

    # --- characterize into impacts
    lcia_rows, lcia_totals, lcia_by_stage = characterize_multi(
        lci_all,
        ef_table,
        annual_output_kg
    )

    # --- optional normalization
    norm_df = pd.read_excel(xls, sheet_name="Normalization") if "Normalization" in xls.sheet_names else None
    norm_totals, norm_by_stage = maybe_normalize(lcia_totals, lcia_by_stage, norm_df)

    # --- write output workbook
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as xw:
        params_df.to_excel(xw, sheet_name=f"Inputs_{sheet_params[:22]}", index=False)
        derived_tbl.to_excel(xw, sheet_name="Derived_perkg", index=False)
        lci_all.to_excel(xw, sheet_name="LCI_perkg", index=False)
        annual_inventory.to_excel(xw, sheet_name="Annual_inventory", index=False)
        ef_table.to_excel(xw, sheet_name="EF_Defaults", index=False)
        lcia_rows.to_excel(xw, sheet_name="LCIA_rows_multi", index=False)
        lcia_by_stage.to_excel(xw, sheet_name="LCIA_by_stage", index=False)
        lcia_totals.to_excel(xw, sheet_name="LCIA_totals_multi", index=False)
        if norm_totals is not None:
            norm_by_stage.to_excel(xw, sheet_name="Normalized_by_stage", index=False)
            norm_totals.to_excel(xw, sheet_name="Normalized_totals", index=False)

    # --- optional README
    if args.write_readme:
        readme_path = args.readme_path or os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            f"{system}_README.md"
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(README_TEXT)
        print(f"📝 Wrote README to: {readme_path}")

    print(f"✅ Wrote: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()