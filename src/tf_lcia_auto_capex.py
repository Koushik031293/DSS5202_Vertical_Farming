#!/usr/bin/env python3
# tf_lcia_auto_capex.py
#
# Build LCIA workbook for TRADITIONAL FARM (open field / conventional),
# aligned with vf_lcia_auto_capex.py style.
#
# Usage:
# python tf_lcia_auto_capex.py \
#   -i data/input/Traditional_Farm_Template.xlsx \
#   -o data/output/TF_LCIA_ready_multiimpact.xlsx \
#   --prefer-split-electricity --prefer-water-m3 \
#   --write-readme
#
# Output sheets:
#   Parameters
#   CAPEX
#   OPEX
#   EF_Defaults
#   Derived_perkg
#   LCI_perkg
#   LCIA_by_stage
#   LCIA_totals_multi
#   LCC
#   README (if --write-readme)
#
# Impact categories:
#   GWP100, HOFP, PMFP, AP, EOFP, FFP
#
# ---------------------------------------------------------------

import argparse
import pandas as pd
import numpy as np

IMPACT_COLS = ["GWP100","HOFP","PMFP","AP","EOFP","FFP"]

# ---------- sheet loading / parameter handling ----------

def load_sheet(xls_path, sheet, required=False):
    try:
        return pd.read_excel(xls_path, sheet_name=sheet)
    except Exception as e:
        if required:
            raise RuntimeError(f"Required sheet '{sheet}' not found in {xls_path}") from e
        return pd.DataFrame()

def params_to_dict(df_params):
    out = {}
    for _, r in df_params.iterrows():
        key = str(r.get("Parameter","")).strip()
        if not key:
            continue
        out[key] = r.get("Value", np.nan)
    return out

# ---------- finance / costing logic ----------

def annualized_capex_row(price, salvage, life_yrs, r):
    price   = 0.0 if pd.isna(price)   else float(price)
    salvage = 0.0 if pd.isna(salvage) else float(salvage)
    life    = 1.0 if pd.isna(life_yrs) or life_yrs <= 0 else float(life_yrs)
    r       = 0.0 if pd.isna(r)       else float(r)

    if r == 0:
        return (price - salvage) / life

    crf = (r * (1+r)**life) / ((1+r)**life - 1)
    return (price - salvage) * crf + salvage * r

def compute_capex_perkg(df_capex, params):
    total_output = params.get("Annual edible output (kg/yr)", np.nan)
    r            = params.get("Discount rate r (decimal)", 0.08)

    df_capex = df_capex.copy()

    for col in ["Price_SGD","Lifetime_years","Salvage_SGD"]:
        if col not in df_capex.columns:
            df_capex[col] = 0.0

    if "Annualized_SGD" not in df_capex.columns:
        df_capex["Annualized_SGD"] = np.nan

    df_capex["Annualized_SGD"] = df_capex.apply(
        lambda row: annualized_capex_row(
            row.get("Price_SGD",0.0),
            row.get("Salvage_SGD",0.0),
            row.get("Lifetime_years",1.0),
            r
        ) if pd.isna(row.get("Annualized_SGD", np.nan)) else row.get("Annualized_SGD"),
        axis=1
    )

    annual_capex_total = df_capex["Annualized_SGD"].sum()
    perkg_capex = (annual_capex_total / total_output
                   if total_output and total_output != 0 else np.nan)

    return perkg_capex, annual_capex_total, df_capex

def compute_opex(df_opex, total_output):
    df = df_opex.copy()

    for col in ["Item","Qty_per_kg","Unit_price_SGD","Cost_per_kg_SGD","Unit"]:
        if col not in df.columns:
            # numeric-like cols default 0, others ""
            if ("qty" in col.lower() or
                "price" in col.lower() or
                "cost" in col.lower()):
                df[col] = 0.0
            else:
                df[col] = ""

    missing_mask = df["Cost_per_kg_SGD"].isna()
    df.loc[missing_mask,"Cost_per_kg_SGD"] = (
        df.loc[missing_mask,"Qty_per_kg"].astype(float) *
        df.loc[missing_mask,"Unit_price_SGD"].astype(float)
    )

    perkg_opex = df["Cost_per_kg_SGD"].sum()
    annual_opex_total = perkg_opex * total_output if total_output else np.nan

    return perkg_opex, annual_opex_total, df

def build_derived_perkg(params_dict, df_opex_fixed):
    """
    Output MUST always have two columns exactly:
    'Item' and 'Value'.
    We'll infer rates from df_opex_fixed["Item"] fuzzy match.
    If df_opex_fixed["Item"] is missing or empty, we still output rows with 0.
    """
    total_output = params_dict.get("Annual edible output (kg/yr)", np.nan)

    def pull_qty(substr):
        if "Item" in df_opex_fixed.columns:
            mask = df_opex_fixed["Item"].astype(str).str.lower().str.contains(
                substr.lower(), na=False
            )
            if mask.any():
                return float(df_opex_fixed.loc[mask, "Qty_per_kg"].iloc[0])
        return 0.0

    rows = [
        ("Annual output (kg/yr)", total_output),
        ("Electricity (kWh/kg)", pull_qty("electric")),
        ("Water (m3/kg)",       pull_qty("water")),      # "Water supply"
        ("Sewer (m3/kg)",       pull_qty("wastewater")),
        ("Fertilizer (kg/kg)",  pull_qty("fertilizer")),
        ("Pesticide (kg/kg)",   pull_qty("pesticide")),
    ]

    df_out = pd.DataFrame(rows, columns=["Item","Value"])
    # Enforce correct dtypes / no missing columns
    if "Item" not in df_out.columns:
        df_out["Item"] = ["Annual output (kg/yr)",
                          "Electricity (kWh/kg)",
                          "Water (m3/kg)",
                          "Sewer (m3/kg)",
                          "Fertilizer (kg/kg)",
                          "Pesticide (kg/kg)"]
    if "Value" not in df_out.columns:
        df_out["Value"] = [total_output,0,0,0,0,0]
    return df_out

# ---------- LCIA logic ----------

def infer_stage(flow_name: str) -> str:
    f = flow_name.lower()
    if "electric"   in f: return "Electricity"
    if "water"      in f and "tap" in f: return "Water"
    if "sewer"      in f or "wastewater" in f: return "Sewerage"
    if "fertilizer" in f or "npk" in f: return "Fertilizer"
    if "pesticide"  in f: return "Pesticide"
    return "Other"

def infer_category(flow_name: str) -> str:
    f = flow_name.lower()
    if "electric"   in f: return "energy"
    if "water"      in f: return "water"
    if "sewer"      in f or "waste" in f: return "water"
    if "fertilizer" in f or "pesticide" in f: return "agrochemicals"
    return "other"

def _safe_get_columns_ef(df_ef):
    """
    Normalize EF_Defaults columns so we always have:
    flow_name, unit, and IMPACT_COLS.
    """
    cand_flow = ["flow_name","Flow","FLOW","Process","Activity","Flow name","Name"]
    cand_unit = ["unit","Unit","Quantity unit","Qty unit","Units"]

    def pick(candidates, columns):
        for c in candidates:
            if c in columns:
                return c
        return None

    flow_col = pick(cand_flow, df_ef.columns)
    unit_col = pick(cand_unit, df_ef.columns)

    norm = pd.DataFrame()

    if flow_col is not None:
        norm["flow_name"] = df_ef[flow_col].astype(str).fillna("")
    else:
        norm["flow_name"] = df_ef.index.astype(str)

    if unit_col is not None:
        norm["unit"] = df_ef[unit_col].astype(str).fillna("")
    else:
        norm["unit"] = ""

    for imp in IMPACT_COLS:
        if imp in df_ef.columns:
            norm[imp] = pd.to_numeric(df_ef[imp], errors="coerce").fillna(0.0)
        else:
            norm[imp] = 0.0

    return norm

def build_lci_perkg(df_derived, df_ef_raw):
    """
    Build per-kg inventory + impact.

    df_derived:
        columns ["Item","Value"].
        Item strings like 'Electricity (kWh/kg)', etc.
    df_ef_raw:
        EF_Defaults original.
    """

    df_ef = _safe_get_columns_ef(df_ef_raw)

    # We'll create a safe lookup dict from df_derived,
    # even if df_derived doesn't have "Item" due to weird edits.
    lookup_list = []
    if "Item" in df_derived.columns and "Value" in df_derived.columns:
        for _, rr in df_derived.iterrows():
            key = str(rr.get("Item","")).lower()
            val = rr.get("Value", 0.0)
            try:
                val = float(val)
            except:
                val = 0.0
            lookup_list.append((key,val))
    else:
        # fallback to empty list
        lookup_list = []

    def fuzzy_lookup(partial):
        """
        Find first match whose key contains the substring 'partial'.
        Return 0.0 if nothing matches.
        """
        p = partial.lower()
        for k,v in lookup_list:
            if p in k:
                return v
        return 0.0

    records = []
    for _, r in df_ef.iterrows():
        flow = str(r.get("flow_name","")).strip()
        unit = r.get("unit","")

        f_low = flow.lower()

        # Map EF flow -> activity intensity per kg using fuzzy_lookup
        if "electric" in f_low:
            amt = fuzzy_lookup("electric")
        elif "water" in f_low and "tap" in f_low:
            # try water m3/kg first, then generic water
            amt = fuzzy_lookup("water (m3")
            if amt == 0.0:
                amt = fuzzy_lookup("water")
        elif "sewer" in f_low or "wastewater" in f_low:
            amt = fuzzy_lookup("sewer")
        elif "fertilizer" in f_low or "npk" in f_low:
            amt = fuzzy_lookup("fertilizer")
        elif "pesticide" in f_low:
            amt = fuzzy_lookup("pesticide")
        else:
            amt = 0.0

        stage = infer_stage(flow)

        out = {
            "flow_name": flow,
            "category": infer_category(flow),
            "unit": unit,
            "amount_per_kg": amt,
            "stage": stage,
        }

        for imp in IMPACT_COLS:
            factor = 0.0 if pd.isna(r.get(imp, np.nan)) else float(r.get(imp))
            out[imp] = factor
            out[f"{imp}_perkg"] = amt * factor

        records.append(out)

    df_lci = pd.DataFrame(records)
    return df_lci

def build_lcia_by_stage(df_lci):
    agg_cols = [f"{c}_perkg" for c in IMPACT_COLS]
    out = df_lci.groupby("stage", dropna=False)[agg_cols].sum().reset_index()
    return out

def build_lcia_totals_multi(df_stage, total_output):
    recs = []
    for imp in IMPACT_COLS:
        col = f"{imp}_perkg"
        perkg_val = df_stage[col].sum() if col in df_stage.columns else 0.0
        peryr_val = perkg_val * total_output if total_output else np.nan
        recs.append({
            "Impact": imp,
            "Total_per_kg": perkg_val,
            "Total_per_year": peryr_val
        })
    return pd.DataFrame(recs)

# ---------- LCC summary ----------

def build_lcc_df(params, perkg_capex, perkg_opex,
                 annual_capex_total, annual_opex_total):
    total_output = params.get("Annual edible output (kg/yr)", np.nan)

    possible_price = None
    for k, v in params.items():
        lk = k.lower()
        if "price" in lk and "kg" in lk:
            possible_price = v
            break
    if possible_price is None:
        possible_price = 2.0  # fallback assumption

    lcov = (perkg_capex or 0.0) + (perkg_opex or 0.0)
    annual_revenue = possible_price * total_output if total_output else np.nan
    annual_cost    = lcov * total_output if total_output else np.nan
    annual_profit  = annual_revenue - annual_cost

    rows = [
        ["Per-kg CAPEX (SGD/kg)", perkg_capex,          "SGD/kg", "Annualized CAPEX / output"],
        ["Per-kg OPEX (SGD/kg)",  perkg_opex,           "SGD/kg", "Sum variable costs per kg"],
        ["LCOv (SGD/kg)",         lcov,                 "SGD/kg", "CAPEX/kg + OPEX/kg"],
        ["Annual CAPEX SGD/yr",   annual_capex_total,   "SGD/yr", "Sum Annualized_SGD"],
        ["Annual OPEX SGD/yr",    annual_opex_total,    "SGD/yr", "OPEX/kg * output"],
        ["Annual output (kg/yr)", total_output,         "kg/yr",  "From Parameters"],
        ["Assumed sell price (SGD/kg)", possible_price, "SGD/kg", "From Parameters or default=2"],
        ["Annual revenue (SGD/yr)", annual_revenue,     "SGD/yr", "price * output"],
        ["Annual cost (SGD/yr)",   annual_cost,         "SGD/yr", "LCOv * output"],
        ["Annual profit (SGD/yr)", annual_profit,       "SGD/yr", "revenue - cost"],
    ]
    return pd.DataFrame(rows, columns=["Metric","Value","Units","Notes"])

# ---------- README sheet builder ----------

def build_readme(params, split_elec_flag, water_pref_flag):
    txt = []
    txt.append("Traditional Farm LCIA build notes")
    txt.append("")
    txt.append(f"- Annual edible output (kg/yr): {params.get('Annual edible output (kg/yr)', 'NA')}")
    txt.append(f"- Discount rate r (decimal): {params.get('Discount rate r (decimal)', 'NA')}")
    txt.append("")
    txt.append("CLI flags (kept for parity with VF):")
    txt.append(f"  --prefer-split-electricity = {split_elec_flag}")
    txt.append(f"  --prefer-water-m3          = {water_pref_flag}")
    txt.append("")
    txt.append("Pipeline:")
    txt.append("1. Annualize CAPEX by CRF => per-kg CAPEX.")
    txt.append("2. Sum OPEX cost/kg => per-kg OPEX.")
    txt.append("3. LCOv = CAPEX/kg + OPEX/kg.")
    txt.append("4. Derive activity intensities per kg (kWh/kg, m3/kg, fertilizer kg/kg).")
    txt.append("5. Multiply activity by EF_Defaults factors to get impact per kg.")
    txt.append("6. Multiply per-kg impact by annual kg/yield to get annual totals.")
    txt.append("")
    txt.append("Impact categories:")
    txt.append(" - GWP100  (kg CO2-eq)")
    txt.append(" - HOFP    (kg NMVOC-eq)")
    txt.append(" - PMFP    (kg PM2.5-eq)")
    txt.append(" - AP      (kg SO2-eq)")
    txt.append(" - EOFP    (kg PO4-eq)")
    txt.append(" - FFP     (MJ fossil resource)")
    return pd.DataFrame({"README": txt})

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Auto-build LCIA workbook for a traditional farm.")
    parser.add_argument("-i","--input", required=True,
                        help="Input Traditional_Farm_Template.xlsx")
    parser.add_argument("-o","--output", required=True,
                        help="Output workbook path (will overwrite/create)")
    parser.add_argument("--prefer-split-electricity", action="store_true",
                        help="CLI compatibility with VF; not critical here")
    parser.add_argument("--prefer-water-m3", action="store_true",
                        help="CLI compatibility with VF; not critical here")
    parser.add_argument("--write-readme", action="store_true",
                        help="If set, add a README sheet with method notes")

    args = parser.parse_args()

    inp  = args.input
    outp = args.output

    # Load core sheets
    df_params = load_sheet(inp, "Parameters", required=True)
    df_capex  = load_sheet(inp, "CAPEX", required=True)
    df_opex   = load_sheet(inp, "OPEX", required=True)
    df_ef     = load_sheet(inp, "EF_Defaults", required=True)

    # Params dict
    params = params_to_dict(df_params)
    total_output = params.get("Annual edible output (kg/yr)", np.nan)

    # CAPEX
    perkg_capex, annual_capex_total, df_capex_fixed = compute_capex_perkg(df_capex, params)

    # OPEX
    perkg_opex, annual_opex_total, df_opex_fixed = compute_opex(df_opex, total_output)

    # Derived_perkg
    df_derived = build_derived_perkg(params, df_opex_fixed)

    # LCI_perkg
    df_lci = build_lci_perkg(df_derived, df_ef)

    # LCIA_by_stage and totals
    df_stage  = build_lcia_by_stage(df_lci)
    df_totals = build_lcia_totals_multi(df_stage, total_output)

    # LCC summary
    df_lcc = build_lcc_df(params,
                          perkg_capex,
                          perkg_opex,
                          annual_capex_total,
                          annual_opex_total)

    # Optional README
    df_readme = None
    if args.write_readme:
        df_readme = build_readme(params,
                                 args.prefer_split_electricity,
                                 args.prefer_water_m3)

    # Write output workbook
    with pd.ExcelWriter(outp, engine="openpyxl") as writer:
        df_params.to_excel(writer, sheet_name="Parameters", index=False)
        df_capex_fixed.to_excel(writer, sheet_name="CAPEX", index=False)
        df_opex_fixed.to_excel(writer, sheet_name="OPEX", index=False)
        df_ef.to_excel(writer, sheet_name="EF_Defaults", index=False)

        df_derived.to_excel(writer, sheet_name="Derived_perkg", index=False)
        df_lci.to_excel(writer,     sheet_name="LCI_perkg", index=False)
        df_stage.to_excel(writer,   sheet_name="LCIA_by_stage", index=False)
        df_totals.to_excel(writer,  sheet_name="LCIA_totals_multi", index=False)
        df_lcc.to_excel(writer,     sheet_name="LCC", index=False)

        if df_readme is not None:
            df_readme.to_excel(writer, sheet_name="README", index=False)

    lcov_val = ((perkg_capex or 0.0) + (perkg_opex or 0.0))
    print(f"✅ Done. Wrote {outp}")
    if perkg_capex == perkg_capex:
        print(f"- Per-kg CAPEX (SGD/kg): {perkg_capex:.4f}")
    else:
        print(f"- Per-kg CAPEX (SGD/kg): NaN")
    if perkg_opex == perkg_opex:
        print(f"- Per-kg OPEX  (SGD/kg): {perkg_opex:.4f}")
    else:
        print(f"- Per-kg OPEX  (SGD/kg): NaN")
    print(f"- LCOv (SGD/kg):         {lcov_val:.4f}")

if __name__ == "__main__":
    main()