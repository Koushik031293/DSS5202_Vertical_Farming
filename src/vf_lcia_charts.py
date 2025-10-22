#!/usr/bin/env python3
# vf_lcia_charts.py
# Generate LCIA charts and compile them into a single Markdown report.

import argparse, os, re, glob
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
    if "category" not in df.columns:
        return
    df = df.set_index("category").reindex(CATS).dropna()
    if df.empty:
        return
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
        large.loc["Other"] = float(small.sum())
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

# --------------------------
# Markdown report generator
# --------------------------
def _exists(path): 
    return os.path.exists(path)

def _rel(path_from, to):
    # return relative path from Markdown file location to target "to"
    return os.path.relpath(to, start=os.path.dirname(path_from))

def build_markdown(outdir, fmt, md_path, title="Vertical Farming LCIA – Charts"):
    # Collect chart files
    expect = [
        ("Total Impact per Category (per kg)",           f"{outdir}/totals_bar.{fmt}"),
        ("LCIA by Stage (stacked per category)",         f"{outdir}/by_stage_stacked_bar.{fmt}"),
        ("GWP100 by Stage (per kg) – Pie/Bar",           f"{outdir}/gwp_stage_pie.{fmt}" if _exists(f"{outdir}/gwp_stage_pie.{fmt}") else f"{outdir}/gwp_stage_bar.{fmt}"),
        ("Normalized Impacts (per kg) – Radar",          f"{outdir}/normalized_radar.{fmt}"),
        ("Top flows by GWP100 (per kg)",                 f"{outdir}/top_flows_gwp100.{fmt}"),
        ("Normalized LCIA by Stage (stacked 0–1)",       f"{outdir}/normalized_by_stage_stacked.{fmt}"),
    ]
    # Per-category stage bars
    per_cat = sorted(glob.glob(os.path.join(outdir, f"by_stage_*_bar.{fmt}")))
    # Build markdown
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("> This report is auto-generated from the LCIA workbook and includes every chart found in the output directory.")
    lines.append("")
    # TOC
    lines.append("## Contents")
    toc_items = [
        ("totals", "Total Impact per Category"),
        ("stacked", "LCIA by Stage (Stacked)"),
        ("gwp", "GWP100 by Stage (Pie/Bar)"),
        ("radar", "Normalized Radar (if available)"),
        ("topflows", "Top Flows by GWP100"),
    ]
    if any("normalized_by_stage_stacked" in p for _, p in expect if p):
        toc_items.append(("normstage", "Normalized by Stage (Stacked 0–1)"))
    if per_cat:
        toc_items.append(("percat", "Per-Category Stage Bars"))
    for anchor, text in toc_items:
        lines.append(f"- [{text}](#{anchor})")
    lines.append("")

    # Sections
    # 1
    lines.append("## Total Impact per Category {#totals}")
    if _exists(expect[0][1]):
        rel = _rel(md_path, expect[0][1])
        lines.append(f"![Total impact per category]({rel})")
    else:
        lines.append("_Chart not generated._")
    lines.append("")

    # 2
    lines.append("## LCIA by Stage (Stacked) {#stacked}")
    if _exists(expect[1][1]):
        rel = _rel(md_path, expect[1][1])
        lines.append(f"![LCIA by Stage (stacked)]({rel})")
    else:
        lines.append("_Chart not generated._")
    lines.append("")

    # 3
    lines.append("## GWP100 by Stage (Pie/Bar) {#gwp}")
    gwp_path = expect[2][1]
    if _exists(gwp_path):
        rel = _rel(md_path, gwp_path)
        lines.append(f"![GWP100 by stage]({rel})")
    else:
        lines.append("_Chart not generated._")
    lines.append("")

    # 4
    lines.append("## Normalized Radar (if available) {#radar}")
    if _exists(expect[3][1]):
        rel = _rel(md_path, expect[3][1])
        lines.append(f"![Normalized radar]({rel})")
    else:
        lines.append("_Chart not generated or normalization sheet missing._")
    lines.append("")

    # 5
    lines.append("## Top Flows by GWP100 {#topflows}")
    if _exists(expect[4][1]):
        rel = _rel(md_path, expect[4][1])
        lines.append(f"![Top flows by GWP100]({rel})")
    else:
        lines.append("_Chart not generated._")
    lines.append("")

    # 6 (optional)
    if _exists(expect[5][1]):
        lines.append("## Normalized by Stage (Stacked 0–1) {#normstage}")
        rel = _rel(md_path, expect[5][1])
        lines.append(f"![Normalized by Stage (stacked 0–1)]({rel})")
        lines.append("")

    # 7 per-category
    if per_cat:
        lines.append("## Per-Category Stage Bars {#percat}")
        lines.append("_One chart per impact category showing per-stage values._")
        lines.append("")
        # Sort in CATS order if possible
        def cat_key(p):
            m = re.search(r"by_stage_(.+?)_bar\.", os.path.basename(p))
            if not m: return (999, p)
            cat = m.group(1)
            try:
                return (CATS.index(cat), p)
            except ValueError:
                return (998, p)
        per_cat_sorted = sorted(per_cat, key=cat_key)
        for p in per_cat_sorted:
            cat = re.search(r"by_stage_(.+?)_bar\.", os.path.basename(p))
            cat = cat.group(1) if cat else "Category"
            rel = _rel(md_path, p)
            lines.append(f"### {cat}")
            lines.append(f"![{cat} by Stage]({rel})")
            lines.append("")
    # Write file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser(description="Generate LCIA charts from VF_LCIA_ready_multiimpact.xlsx and compile a Markdown report.")
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

    # New: Markdown options
    ap.add_argument("--write-md", action="store_true", help="Write a Markdown report that embeds all available charts")
    ap.add_argument("--md-file", default="LCIA_charts_report.md", help="Markdown file path to write")
    ap.add_argument("--md-title", default="Vertical Farming LCIA – Charts", help="Title for the Markdown report")

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

    # Build Markdown if requested
    if args.write_md:
        md_path = args.md_file
        # If user passes only a filename, place it next to outdir for easy linking
        if not os.path.isabs(md_path):
            md_path = os.path.join(os.getcwd(), md_path)
        build_markdown(args.outdir, args.fmt, md_path, title=args.md_title)

    print(f"✅ Charts written to: {os.path.abspath(args.outdir)}")
    if args.write_md:
        print(f"📝 Markdown report: {os.path.abspath(args.md_file)}")

if __name__ == "__main__":
    main()