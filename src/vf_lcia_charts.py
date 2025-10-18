#!/usr/bin/env python3
# vf_lcia_charts.py
# Generate LCIA charts from the workbook produced by vf_lcia_auto_capex.py
#
# Charts produced (saved to --outdir):
# 1) totals_bar.<ext>                      - Total impact per category (per kg)
# 2) by_stage_stacked_bar.<ext>            - Stacked bar of stages for each category (per kg) [multi-unit]
# 3) gwp_stage_pie.<ext> or gwp_stage_bar.<ext> (auto) - GWP100 by stage
# 4) normalized_radar.<ext>                - Radar chart of normalized totals (if Normalized_totals exists)
# 5) top_flows_gwp100.<ext>                - Horizontal bar: top-N flows by GWP100 (per kg)
# 6) by_stage_<CAT>_bar.<ext> (one per CAT) - Category-specific per-stage bars
# 7) normalized_by_stage_stacked.<ext>     - Stacked (0–1) to compare stages across categories
#
# Usage:
# python vf_lcia_charts.py -i VF_LCIA_ready_multiimpact.xlsx -o charts --fmt png --dpi 160 --topn 10 \
#   --per-cat --logy-per-cat --normalized-by-stage

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CATS = ["GWP100","HOFP","PMFP","AP","EOFP","FFP"]

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def read_sheet(xlsx, name):
    try:
        return pd.read_excel(xlsx, sheet_name=name)
    except Exception:
        return None

def chart_totals_bar(totals_df, outpath, fmt="png", dpi=160):
    df = totals_df.copy()
    if "perkg_total" not in df.columns and "perkg" in df.columns:
        df["perkg_total"] = df["perkg"]
    df = df.set_index("category").reindex(CATS).dropna()
    plt.figure(figsize=(7,4))
    df["perkg_total"].plot(kind="bar")
    plt.ylabel("Impact per kg (category units)")
    plt.title("Total Impact per Category (per kg)")
    plt.tight_layout()
    plt.savefig(f"{outpath}/totals_bar.{fmt}", dpi=dpi)
    plt.close()

def chart_by_stage_stacked(by_stage_df, outpath, fmt="png", dpi=160):
    df = by_stage_df.copy()
    if "stage" not in df.columns:
        return
    df = df.set_index("stage")
    cols = [c for c in CATS if c in df.columns]
    if not cols:
        return
    plt.figure(figsize=(10,5))
    df[cols].plot(kind="bar", stacked=True)
    plt.ylabel("Impact per kg (category units)")
    plt.title("LCIA by Stage (stacked per category)")
    plt.tight_layout()
    plt.savefig(f"{outpath}/by_stage_stacked_bar.{fmt}", dpi=dpi)
    plt.close()

def _pie_group_small(s, min_slice_pct=0.02):
    total = float(s.sum())
    if total <= 0:
        return s
    share = s / total
    large = s[share >= min_slice_pct]
    small = s[share < min_slice_pct]
    if small.sum() > 0:
        large.loc["Other"] = small.sum()
    return large.sort_values(ascending=False)

def chart_gwp_stage_pie_or_bar(by_stage_df, outpath, fmt="png", dpi=160, dominance_threshold=0.9, min_slice_pct=0.02, legend_pie=True):
    df = by_stage_df.copy()
    if "stage" not in df.columns or "GWP100" not in df.columns:
        return
    s = df.set_index("stage")["GWP100"]
    s = s[s > 0]
    if s.empty:
        return
    s = s.sort_values(ascending=False)
    total = float(s.sum())

    if s.iloc[0] >= dominance_threshold * total:
        plt.figure(figsize=(7,4))
        s.sort_values().plot(kind="barh")
        plt.xlabel("GWP100 (kg CO₂e per kg)")
        plt.title("GWP100 by Stage (per kg)")
        plt.tight_layout()
        plt.savefig(f"{outpath}/gwp_stage_bar.{fmt}", dpi=dpi)
        plt.close()
        return

    sg = _pie_group_small(s, min_slice_pct=min_slice_pct)
    labels = list(sg.index); values = sg.values
    explode = [0.1 if i < 3 else 0 for i in range(len(values))]

    def _autopct(p): return f"{p:.1f}%" if p >= 1 else ""

    fig, ax = plt.subplots(figsize=(7,7))
    wedges, texts, autotexts = ax.pie(values,
                                      explode=explode,
                                      startangle=90,
                                      autopct=_autopct,
                                      pctdistance=0.8,
                                      labels=None if legend_pie else labels,
                                      labeldistance=1.1 if not legend_pie else 1.05)
    ax.set_title("GWP100 by Stage (per kg)")
    if legend_pie:
        pct = 100 * (sg / sg.sum())
        legend_labels = [f"{lab} — {p:.1f}%" for lab, p in zip(labels, pct)]
        ax.legend(wedges, legend_labels, title="Stage", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(f"{outpath}/gwp_stage_pie.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def chart_normalized_radar(norm_totals_df, outpath, fmt="png", dpi=160):
    df = norm_totals_df.copy()
    if df is None or df.empty:
        return
    if "category" not in df.columns or "normalized_perkg" not in df.columns:
        return
    s = df.set_index("category").reindex(CATS)["normalized_perkg"].fillna(0.0)
    labels = list(s.index); values = s.values.tolist()
    labels += labels[:1]; values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]
    fig = plt.figure(figsize=(6.5,6.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values); ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels[:-1])
    ax.set_title("Normalized Impacts (per kg)")
    plt.tight_layout(); plt.savefig(f"{outpath}/normalized_radar.{fmt}", dpi=dpi); plt.close(fig)

def chart_top_flows(lcia_rows_df, outpath, fmt="png", dpi=160, topn=10, category="GWP100"):
    df = lcia_rows_df.copy()
    perkg_col = f"{category}_perkg"
    if perkg_col not in df.columns or "flow_name" not in df.columns:
        return
    s = df.groupby("flow_name")[perkg_col].sum().sort_values(ascending=False).head(topn)
    if s.empty: return
    plt.figure(figsize=(8,6))
    s.sort_values().plot(kind="barh")
    plt.xlabel(f"{category} per kg")
    plt.title(f"Top {len(s)} Flows by {category} (per kg)")
    plt.tight_layout(); plt.savefig(f"{outpath}/top_flows_{category.lower()}.{fmt}", dpi=dpi); plt.close()

def chart_per_category_stage_bars(by_stage_df, outpath, fmt="png", dpi=160, logy=False):
    """Generate a separate bar chart for each category vs stage."""
    df = by_stage_df.copy()
    if "stage" not in df.columns: return
    df = df.set_index("stage")
    for cat in [c for c in CATS if c in df.columns]:
        plt.figure(figsize=(7,4))
        df[cat].plot(kind="bar")
        if logy: plt.yscale("log")
        plt.ylabel(cat)
        plt.title(f"{cat} by Stage (per kg)")
        plt.tight_layout()
        plt.savefig(f"{outpath}/by_stage_{cat}_bar.{fmt}", dpi=dpi)
        plt.close()

def chart_normalized_by_stage(by_stage_df, outpath, fmt="png", dpi=160):
    """Stacked plot with each category normalized to its column max (0–1)."""
    df = by_stage_df.copy()
    if "stage" not in df.columns: return
    df = df.set_index("stage")
    cols = [c for c in CATS if c in df.columns]
    if not cols: return
    norm = df[cols].div(df[cols].max(axis=0), axis=1).fillna(0.0)
    plt.figure(figsize=(10,5))
    norm.plot(kind="bar", stacked=True)
    plt.ylim(0,1)
    plt.ylabel("Relative impact (0–1 within category)")
    plt.title("Normalized LCIA by Stage (per category scaled to max=1)")
    plt.tight_layout()
    plt.savefig(f"{outpath}/normalized_by_stage_stacked.{fmt}", dpi=dpi)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Generate LCIA charts from VF_LCIA_ready_multiimpact.xlsx")
    ap.add_argument("-i","--input", required=True, help="Path to LCIA workbook")
    ap.add_argument("-o","--outdir", default="charts", help="Directory to save charts")
    ap.add_argument("--fmt", default="png", choices=["png","pdf","svg"], help="Output image format")
    ap.add_argument("--dpi", type=int, default=160, help="DPI for raster formats")
    ap.add_argument("--topn", type=int, default=10, help="Top N flows for hotspot chart")
    ap.add_argument("--dominance-threshold", type=float, default=0.9, help="Max share for pie; else use bar")
    ap.add_argument("--min-slice-pct", type=float, default=0.02, help="Group slices smaller than this into 'Other'")
    ap.add_argument("--legend-pie", action="store_true", help="Use legend instead of outer labels for pie")
    ap.add_argument("--per-cat", action="store_true", help="Generate category-specific per-stage bars")
    ap.add_argument("--logy-per-cat", action="store_true", help="Use log scale on per-category charts")
    ap.add_argument("--normalized-by-stage", action="store_true", help="Generate normalized-by-stage stacked chart")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    totals = read_sheet(args.input, "LCIA_totals_multi")
    by_stage = read_sheet(args.input, "LCIA_by_stage")
    rows = read_sheet(args.input, "LCIA_rows_multi")
    norm_totals = read_sheet(args.input, "Normalized_totals")

    if totals is not None and not totals.empty and "category" in totals.columns:
        chart_totals_bar(totals, args.outdir, args.fmt, args.dpi)

    if by_stage is not None and not by_stage.empty:
        chart_by_stage_stacked(by_stage, args.outdir, args.fmt, args.dpi)
        chart_gwp_stage_pie_or_bar(by_stage, args.outdir, args.fmt, args.dpi,
                                   dominance_threshold=args.dominance_threshold,
                                   min_slice_pct=args.min_slice_pct,
                                   legend_pie=args.legend_pie)
        if args.per_cat:
            chart_per_category_stage_bars(by_stage, args.outdir, args.fmt, args.dpi, logy=args.logy_per_cat)
        if args.normalized_by_stage:
            chart_normalized_by_stage(by_stage, args.outdir, args.fmt, args.dpi)

    if norm_totals is not None and not norm_totals.empty:
        chart_normalized_radar(norm_totals, args.outdir, args.fmt, args.dpi)

    if rows is not None and not rows.empty:
        chart_top_flows(rows, args.outdir, args.fmt, args.dpi, args.topn, category="GWP100")

    print(f"✅ Charts written to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
