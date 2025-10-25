#!/usr/bin/env python3
# lcia_charts_unified.py
# Generate LCIA charts (VF/TF compatible) + Markdown/PDF.
# Now supports --label so filenames are unique per scenario (e.g. VF_, TF_).

import argparse, os, re, glob, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CATS = ["GWP100","HOFP","PMFP","AP","EOFP","FFP"]

def safe_mkdir(path): os.makedirs(path, exist_ok=True)
def _exists(path): return os.path.exists(path)
def _rel(path_from, to): return os.path.relpath(to, start=os.path.dirname(path_from))

def read_sheet(xlsx, name):
    try:
        return pd.read_excel(xlsx, sheet_name=name)
    except Exception:
        return None

# ---------- normalization helpers ----------
def normalize_totals_df(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["category","perkg_total"])
    df = raw.copy()
    if "category" in df.columns and "perkg_total" in df.columns:
        out = df[["category","perkg_total"]].copy()
        return out
    if "Impact" in df.columns and "Total_per_kg" in df.columns:
        out = df.rename(columns={"Impact":"category","Total_per_kg":"perkg_total"})[
            ["category","perkg_total"]
        ].copy()
        return out
    # fallback guess
    str_col = None
    num_col = None
    for c in df.columns:
        if df[c].dtype == object and str_col is None:
            str_col = c
        if pd.api.types.is_numeric_dtype(df[c]) and num_col is None:
            num_col = c
    if str_col and num_col:
        out = df[[str_col,num_col]].copy()
        out.columns = ["category","perkg_total"]
        return out
    return pd.DataFrame(columns=["category","perkg_total"])

def normalize_by_stage_df(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["stage"]+CATS)
    df = raw.copy()
    if "stage" not in df.columns:
        return pd.DataFrame(columns=["stage"]+CATS)
    norm_cols = {"stage": df["stage"].astype(str)}
    for cat in CATS:
        if cat in df.columns:
            norm_cols[cat] = pd.to_numeric(df[cat], errors="coerce").fillna(0.0)
        elif f"{cat}_perkg" in df.columns:
            norm_cols[cat] = pd.to_numeric(df[f"{cat}_perkg"], errors="coerce").fillna(0.0)
        else:
            norm_cols[cat] = 0.0
    return pd.DataFrame(norm_cols)

def pick_rows_df(lcia_rows_multi, lci_perkg):
    if lcia_rows_multi is not None and not lcia_rows_multi.empty:
        df = lcia_rows_multi.copy()
    elif lci_perkg is not None and not lci_perkg.empty:
        df = lci_perkg.copy()
    else:
        return pd.DataFrame()
    if "flow_name" not in df.columns:
        return pd.DataFrame()
    gwp_col = None
    for c in df.columns:
        c_low = c.lower()
        if "gwp" in c_low and "per" in c_low:
            gwp_col = c
            break
    if gwp_col is None:
        if "GWP100" in df.columns:
            gwp_col = "GWP100"
    if gwp_col is None:
        return pd.DataFrame()
    out = df[["flow_name", gwp_col]].copy()
    out = out.rename(columns={gwp_col: "GWP100_perkg"})
    out["GWP100_perkg"] = pd.to_numeric(out["GWP100_perkg"], errors="coerce").fillna(0.0)
    return out

def normalize_norm_totals_df(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "category" in df.columns and "normalized_perkg" in df.columns:
        return df[["category","normalized_perkg"]].copy()
    return pd.DataFrame()

# ---------- chart helpers ----------
def savefig_with_label(outdir, label_prefix, basename, fmt, dpi=160, bbox_inches=None):
    """
    Centralized save so naming stays consistent.
    label_prefix already includes "_" if you passed it that way.
    """
    fname = os.path.join(outdir, f"{label_prefix}{basename}.{fmt}")
    plt.tight_layout()
    if bbox_inches is not None:
        plt.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
    else:
        plt.savefig(fname, dpi=dpi)
    plt.close()
    return fname

def chart_totals_bar(totals_df_std, outdir, label_prefix, fmt="png", dpi=160):
    df = totals_df_std.copy()
    if "category" not in df.columns or "perkg_total" not in df.columns:
        return None
    df = df.set_index("category").reindex(CATS).dropna()
    if df.empty:
        return None
    plt.figure(figsize=(7,4))
    df["perkg_total"].plot(kind="bar")
    plt.ylabel("Impact per kg (category units)")
    plt.title("Total Impact per Category (per kg)")
    return savefig_with_label(outdir, label_prefix, "totals_bar", fmt, dpi)

def chart_by_stage_stacked(by_stage_std, outdir, label_prefix, fmt="png", dpi=160):
    df = by_stage_std.copy()
    if "stage" not in df.columns:
        return None
    df = df.set_index("stage")
    cols = [c for c in CATS if c in df.columns]
    if not cols:
        return None
    plt.figure(figsize=(10,5))
    df[cols].plot(kind="bar", stacked=True)
    plt.ylabel("Impact per kg (category units)")
    plt.title("LCIA by Stage (stacked per category)")
    return savefig_with_label(outdir, label_prefix, "by_stage_stacked_bar", fmt, dpi)

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

def chart_gwp_stage_pie_or_bar(by_stage_std,
                               outdir,
                               label_prefix,
                               fmt="png",
                               dpi=160,
                               dominance_threshold=0.9,
                               min_slice_pct=0.02,
                               legend_pie=True):
    df = by_stage_std.copy()
    if "stage" not in df.columns or "GWP100" not in df.columns:
        return None, None
    s = df.set_index("stage")["GWP100"]
    s = s[s > 0]
    if s.empty:
        return None, None
    s = s.sort_values(ascending=False)
    total = float(s.sum())

    # Bar fallback if one stage dominates
    if s.iloc[0] >= dominance_threshold * total:
        plt.figure(figsize=(7,4))
        s.sort_values().plot(kind="barh")
        plt.xlabel("GWP100 (kg CO₂e per kg)")
        plt.title("GWP100 by Stage (per kg)")
        bar_path = savefig_with_label(outdir, label_prefix, "gwp_stage_bar", fmt, dpi)
        return bar_path, None

    # Pie chart path
    sg = _pie_group_small(s, min_slice_pct=min_slice_pct)
    labels, values = list(sg.index), sg.values
    explode = [0.1 if i < 3 else 0 for i in range(len(values))]
    def _autopct(p): return f"{p:.1f}%" if p >= 1 else ""

    fig, ax = plt.subplots(figsize=(7,7))
    wedges, texts, autotexts = ax.pie(
        values,
        explode=explode,
        startangle=90,
        autopct=_autopct,
        pctdistance=0.8,
        labels=None if legend_pie else labels,
        labeldistance=1.1 if not legend_pie else 1.05
    )
    ax.set_title("GWP100 by Stage (per kg)")
    if legend_pie:
        pct = 100 * (sg / sg.sum())
        legend_labels = [f"{lab} — {p:.1f}%" for lab, p in zip(labels, pct)]
        ax.legend(
            wedges,
            legend_labels,
            title="Stage",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
    pie_path = savefig_with_label(outdir, label_prefix, "gwp_stage_pie", fmt, dpi, bbox_inches="tight")
    return None, pie_path

def chart_normalized_radar(norm_totals_std, outdir, label_prefix, fmt="png", dpi=160):
    df = norm_totals_std.copy()
    if df is None or df.empty:
        return None
    if "category" not in df.columns or "normalized_perkg" not in df.columns:
        return None
    s = df.set_index("category").reindex(CATS)["normalized_perkg"].fillna(0.0)
    labels = list(s.index)
    values = s.values.tolist()
    labels += labels[:1]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.5,6.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    ax.set_title("Normalized Impacts (per kg)")
    path = savefig_with_label(outdir, label_prefix, "normalized_radar", fmt, dpi)
    return path

def chart_top_flows(rows_std, outdir, label_prefix, fmt="png", dpi=160, topn=10, category="GWP100"):
    if rows_std is None or rows_std.empty:
        return None
    perkg_col = f"{category}_perkg"
    if perkg_col not in rows_std.columns or "flow_name" not in rows_std.columns:
        return None
    s = rows_std.groupby("flow_name")[perkg_col].sum().sort_values(ascending=False).head(topn)
    if s.empty:
        return None
    plt.figure(figsize=(8,6))
    s.sort_values().plot(kind="barh")
    plt.xlabel(f"{category} per kg")
    plt.title(f"Top {len(s)} Flows by {category} (per kg)")
    path = savefig_with_label(outdir, label_prefix, f"top_flows_{category.lower()}", fmt, dpi)
    return path

def chart_per_category_stage_bars(by_stage_std, outdir, label_prefix, fmt="png", dpi=160, logy=False):
    df = by_stage_std.copy()
    if "stage" not in df.columns:
        return []
    df = df.set_index("stage")
    paths = []
    for cat in [c for c in CATS if c in df.columns]:
        plt.figure(figsize=(7,4))
        df[cat].plot(kind="bar")
        if logy:
            plt.yscale("log")
        plt.ylabel(cat)
        plt.title(f"{cat} by Stage (per kg)")
        path = savefig_with_label(outdir, label_prefix, f"by_stage_{cat}_bar", fmt, dpi)
        paths.append(path)
    return paths

def chart_normalized_by_stage(by_stage_std, outdir, label_prefix, fmt="png", dpi=160):
    df = by_stage_std.copy()
    if "stage" not in df.columns:
        return None
    df = df.set_index("stage")
    cols = [c for c in CATS if c in df.columns]
    if not cols:
        return None
    norm = df[cols].div(df[cols].max(axis=0), axis=1).fillna(0.0)
    plt.figure(figsize=(10,5))
    norm.plot(kind="bar", stacked=True)
    plt.ylim(0,1)
    plt.ylabel("Relative impact (0–1 within category)")
    plt.title("Normalized LCIA by Stage (per category scaled to max=1)")
    path = savefig_with_label(outdir, label_prefix, "normalized_by_stage_stacked", fmt, dpi)
    return path

# ---------- Markdown builder ----------
def build_markdown(outdir, fmt, md_path, title, label_prefix):
    # Figure filenames with label_prefix
    totals_path   = os.path.join(outdir, f"{label_prefix}totals_bar.{fmt}")
    stacked_path  = os.path.join(outdir, f"{label_prefix}by_stage_stacked_bar.{fmt}")
    pie_path      = os.path.join(outdir, f"{label_prefix}gwp_stage_pie.{fmt}")
    bar_path      = os.path.join(outdir, f"{label_prefix}gwp_stage_bar.{fmt}")
    radar_path    = os.path.join(outdir, f"{label_prefix}normalized_radar.{fmt}")
    topflows_path = os.path.join(outdir, f"{label_prefix}top_flows_gwp100.{fmt}")
    normstage_path= os.path.join(outdir, f"{label_prefix}normalized_by_stage_stacked.{fmt}")
    per_cat_paths = sorted(glob.glob(os.path.join(outdir, f"{label_prefix}by_stage_*_bar.{fmt}")))

    lines = [
        f"# {title}",
        "",
        "> Auto-generated LCIA charts.",
        "",
        "## Contents"
    ]

    toc = [
        ("totals","Total Impact per Category"),
        ("stacked","LCIA by Stage (Stacked)"),
        ("gwp","GWP100 by Stage (Pie/Bar)"),
        ("radar","Normalized Radar (if available)"),
        ("topflows","Top Flows by GWP100"),
    ]
    if _exists(normstage_path):
        toc.append(("normstage","Normalized by Stage (Stacked 0–1)"))
    if per_cat_paths:
        toc.append(("percat","Per-Category Stage Bars"))

    lines += [f"- [{text}](#{anchor})" for anchor, text in toc]
    lines.append("")

    # totals
    lines.append("## Total Impact per Category {#totals}")
    rel = _rel(md_path, totals_path) if _exists(totals_path) else None
    lines.append(f"![Total impact per category]({rel})" if rel else "_Chart not generated._")
    lines.append("")

    # stacked
    lines.append("## LCIA by Stage (Stacked) {#stacked}")
    rel = _rel(md_path, stacked_path) if _exists(stacked_path) else None
    lines.append(f"![LCIA by Stage (stacked)]({rel})" if rel else "_Chart not generated._")
    lines.append("")

    # gwp
    lines.append("## GWP100 by Stage (Pie/Bar) {#gwp}")
    chosen_gwp = pie_path if _exists(pie_path) else (bar_path if _exists(bar_path) else None)
    rel = _rel(md_path, chosen_gwp) if chosen_gwp else None
    lines.append(f"![GWP100 by stage]({rel})" if rel else "_Chart not generated._")
    lines.append("")

    # radar
    lines.append("## Normalized Radar (if available) {#radar}")
    rel = _rel(md_path, radar_path) if _exists(radar_path) else None
    lines.append(f"![Normalized radar]({rel})" if rel else "_Chart not generated or normalization sheet missing._")
    lines.append("")

    # top flows
    lines.append("## Top Flows by GWP100 {#topflows}")
    rel = _rel(md_path, topflows_path) if _exists(topflows_path) else None
    lines.append(f"![Top flows by GWP100]({rel})" if rel else "_Chart not generated._")
    lines.append("")

    # normalized by stage stacked
    if _exists(normstage_path):
        lines.append("## Normalized by Stage (Stacked 0–1) {#normstage}")
        rel = _rel(md_path, normstage_path)
        lines.append(f"![Normalized by Stage (stacked 0–1)]({rel})")
        lines.append("")

    # per-category bars
    if per_cat_paths:
        lines.append("## Per-Category Stage Bars {#percat}")
        lines.append("_One chart per impact category showing per-stage values._")
        lines.append("")
        def cat_key(p):
            m = re.search(r"by_stage_(.+?)_bar\.", os.path.basename(p))
            if not m:
                return (999, p)
            cat = m.group(1)
            return ((CATS.index(cat) if cat in CATS else 998), p)
        for p in sorted(per_cat_paths, key=cat_key):
            m = re.search(r"by_stage_(.+?)_bar\.", os.path.basename(p))
            cat = m.group(1) if m else "Category"
            rel = _rel(md_path, p)
            lines.append(f"### {cat}")
            lines.append(f"![{cat} by Stage]({rel})")
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------- PDF helpers ----------
def convert_md_to_pdf(md_file, out_pdf=None, pdf_engine=None):
    import pypandoc
    if out_pdf is None:
        out_pdf = os.path.splitext(md_file)[0] + ".pdf"
    if pdf_engine is None:
        pdf_engine = "xelatex" if shutil.which("xelatex") else ("wkhtmltopdf" if shutil.which("wkhtmltopdf") else None)
    extra = ["--standalone"]
    if pdf_engine:
        extra += [f"--pdf-engine={pdf_engine}"]
    try:
        pypandoc.convert_text(
            open(md_file, encoding="utf-8").read(),
            "pdf",
            format="md",
            outputfile=out_pdf,
            extra_args=extra
        )
        print(f"✅ PDF generated: {out_pdf} (engine={pdf_engine or 'pandoc-default'})")
    except Exception as e:
        print(f"⚠️ PDF conversion failed: {e}")

def images_to_pdf(img_dir, out_pdf):
    from PIL import Image
    imgs = sorted([p for p in glob.glob(os.path.join(img_dir, "*.*"))
                   if os.path.splitext(p)[1].lower() in (".png",".jpg",".jpeg")])
    if not imgs:
        print("⚠️ No images to convert.")
        return
    pages = []
    for p in imgs:
        im = Image.open(p)
        if im.mode in ("RGBA","P"):
            im = im.convert("RGB")
        pages.append(im)
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:])
    print(f"✅ Image-only PDF created: {out_pdf}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate LCIA charts (VF/TF compatible) + Markdown/PDF.")
    ap.add_argument("-i","--input", required=True, help="Path to LCIA workbook (VF or TF output)")
    ap.add_argument("-o","--outdir", default="charts", help="Directory to save charts")

    # NEW: label
    ap.add_argument("--label", default="", help="Short label prefix for filenames, e.g. 'VF' or 'TF'. No spaces.")

    ap.add_argument("--fmt", default="png", choices=["png","pdf","svg"], help="Output image format")
    ap.add_argument("--dpi", type=int, default=160, help="DPI for raster formats")
    ap.add_argument("--topn", type=int, default=10, help="Top N flows for hotspot chart")
    ap.add_argument("--dominance-threshold", type=float, default=0.9, help="If top stage >90%, use bar instead of pie")
    ap.add_argument("--min-slice-pct", type=float, default=0.02, help="Group tiny pie slices into 'Other'")
    ap.add_argument("--legend-pie", action="store_true", help="Use legend instead of labels for GWP pie")
    ap.add_argument("--per-cat", action="store_true", help="Generate per-category per-stage bars")
    ap.add_argument("--logy-per-cat", action="store_true", help="Log scale on per-category charts")
    ap.add_argument("--normalized-by-stage", action="store_true", help="Make normalized-by-stage stacked chart")

    # report options
    ap.add_argument("--write-md", action="store_true", help="Write Markdown report with chart embeds")
    ap.add_argument("--md-file", default="LCIA_charts_report.md", help="Markdown file path")
    ap.add_argument("--md-title", default="LCIA – Charts", help="Markdown title")
    ap.add_argument("--make-pdf", action="store_true", help="Convert Markdown report to PDF (pandoc)")
    ap.add_argument("--pdf-engine", choices=["pdflatex","xelatex","wkhtmltopdf"], help="Force PDF engine")
    ap.add_argument("--images-only-pdf", action="store_true", help="Stitch chart images into 1 PDF (no pandoc)")

    args = ap.parse_args()
    safe_mkdir(args.outdir)

    # normalize label prefix for file names
    label_prefix = args.label.strip()
    if label_prefix != "":
        label_prefix = label_prefix + "_"

    # Read sheets
    totals_raw      = read_sheet(args.input, "LCIA_totals_multi")
    by_stage_raw    = read_sheet(args.input, "LCIA_by_stage")
    rows_raw        = read_sheet(args.input, "LCIA_rows_multi")
    lci_perkg_raw   = read_sheet(args.input, "LCI_perkg")
    norm_totals_raw = read_sheet(args.input, "Normalized_totals")

    # Normalize shapes
    totals_std      = normalize_totals_df(totals_raw)
    by_stage_std    = normalize_by_stage_df(by_stage_raw)
    rows_std        = pick_rows_df(rows_raw, lci_perkg_raw)
    norm_totals_std = normalize_norm_totals_df(norm_totals_raw)

    # Generate charts
    chart_totals_bar(totals_std, args.outdir, label_prefix, args.fmt, args.dpi)

    if not by_stage_std.empty:
        chart_by_stage_stacked(by_stage_std, args.outdir, label_prefix, args.fmt, args.dpi)
        bar_path, pie_path = chart_gwp_stage_pie_or_bar(
            by_stage_std,
            args.outdir,
            label_prefix,
            args.fmt,
            args.dpi,
            dominance_threshold=args.dominance_threshold,
            min_slice_pct=args.min_slice_pct,
            legend_pie=args.legend_pie
        )
        if args.per_cat:
            chart_per_category_stage_bars(by_stage_std, args.outdir, label_prefix, args.fmt, args.dpi, logy=args.logy_per_cat)
        if args.normalized_by_stage:
            chart_normalized_by_stage(by_stage_std, args.outdir, label_prefix, args.fmt, args.dpi)

    chart_normalized_radar(norm_totals_std, args.outdir, label_prefix, args.fmt, args.dpi)

    chart_top_flows(rows_std, args.outdir, label_prefix, args.fmt, args.dpi, args.topn, category="GWP100")

    # Markdown + PDF
    md_path = None
    if args.write_md:
        md_path = os.path.join(os.getcwd(), args.md_file) if not os.path.isabs(args.md_file) else args.md_file
        build_markdown(args.outdir, args.fmt, md_path, args.md_title, label_prefix)
        print(f"📝 Markdown report: {md_path}")
        if args.make_pdf:
            convert_md_to_pdf(md_path, pdf_engine=args.pdf_engine)

    if args.images_only_pdf:
        images_to_pdf(args.outdir, os.path.join(args.outdir, f"{label_prefix}LCIA_charts_images.pdf"))

    print(f"✅ Charts written to: {os.path.abspath(args.outdir)}")
    print(f"   label prefix used: '{label_prefix}'")

if __name__ == "__main__":
    main()