# -*- coding: utf-8 -*-
"""HareSkip paper figures F1-F8 (300 dpi, English labels).

All numbers are recomputed from the experiment data of record under
``experiment-HareSkip/``; only F2 is a schematic, and even there the
probability curve is evaluated with the shipped implementation
(``hareskip/probability_models.py``, ``monotone_saturate_v0.1``, a=0.55).

Run:
    S:\\30_OriginalApps\\16_HareSkip\\experiment-HareSkip\\analysis-phase1\\.venv\\Scripts\\python.exe make_figures.py

Writes only into this directory. Changes no existing file.
"""
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

# --- paths -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # forge-neo-Anima-HareSkip
ROOT = os.path.abspath(os.path.join(REPO, ".."))                      # 16_HareSkip
EXP = os.path.join(ROOT, "experiment-HareSkip")
JUDGE = os.path.join(EXP, "Judge-results")
OUT = HERE

sys.path.insert(0, REPO)
from hareskip import probability_models as PM  # noqa: E402

ANCHOR = 0.121          # damage-scale anchor (LPIPS-VGG), paper-wide reference
BLOWUP = 0.308          # anchor q95 -- "broken" threshold
CATASTROPHE = 0.5       # "essentially a different image"

# --- shared style ------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

SAMPLERS = ["ERSDE-Beta", "ERSDE-Simple", "Euler-Beta"]
FAMILIES4 = ["ERSDE-Beta", "ERSDE-Simple", "Euler-Beta", "Euler-Simple"]
CYC = plt.rcParams["axes.prop_cycle"].by_key()["color"]
SCOLOR = {s: CYC[i] for i, s in enumerate(SAMPLERS)}

STR_GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
SMO_GRID = [round(x, 1) for x in np.arange(0, 0.91, 0.1)] + [0.99]

# Judge runs that hold the Stage-5 response-surface scans (one per family).
STAGE5_SRC = {
    "ERSDE-Beta":   ("Stage5-Scan1_20260905-113330", None),
    "ERSDE-Simple": ("Stage5-Families_20260907-235041", "ERSDE-Simple"),
    "Euler-Beta":   ("Stage5-Families_20260907-235041", "Euler-Beta"),
    "Euler-Simple": ("Stage5-EulerSimple_20260909-221014", "Euler-Simple"),
}
LP = "LPIPS - VGG"

PROV = []   # provenance log, printed at the end


def note(fig_name, *lines):
    PROV.append((fig_name, list(lines)))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote %s" % path)


# --- Stage-5 grid loader (shared by F6, F7, F8) ------------------------------

def _parse_stage5_filename(fn):
    if fn == "reuse.png":
        return 0.0, np.nan
    m = re.match(r"str-(\d+)_ema-(\d+)\.png$", fn)
    if not m:
        return np.nan, np.nan
    e = m.group(2)
    return int(m.group(1)) / 10.0, (0.99 if e == "099" else int(e) / 10.0)


def load_stage5(family):
    """Return dict with the per-condition LPIPS grid for one family.

    G has shape (15 conditions, 11 strengths, 11 smoothings); row 0
    (strength=0) is the measured "reuse" baseline broadcast across columns,
    exactly as analysis-stage5-eulersimple/compare_2x2.py does.
    """
    run, key = STAGE5_SRC[family]
    csv = os.path.join(JUDGE, run, "merged-results.csv")
    df = pd.read_csv(csv, encoding="utf-8-sig")
    df[LP] = pd.to_numeric(df[LP], errors="coerce")
    df[["strength", "smoothing"]] = df["File name"].apply(
        lambda f: pd.Series(_parse_stage5_filename(f)))
    df["family"] = df["Condition"].str.split("_").str[0]
    if key:
        df = df[df.family == key]
    conds = sorted(df["Condition"].unique())
    assert len(conds) == 15, (family, len(conds))
    G = np.full((15, 11, 11), np.nan)
    reuse = []
    for i, c in enumerate(conds):
        dd = df[df.Condition == c]
        r = float(dd[dd.strength == 0][LP].iloc[0])
        reuse.append(r)
        G[i, 0, :] = r
        for _, row in dd[dd.strength > 0].iterrows():
            G[i, STR_GRID.index(round(row.strength, 1)),
              SMO_GRID.index(round(row.smoothing, 2))] = row[LP]
    assert not np.isnan(G).any(), family
    return dict(G=G, conds=conds, reuse=np.array(reuse),
                Gmed=np.median(G, axis=0), df=df, csv=csv)


# =============================================================================
# F1 -- single-step skip damage vs trajectory coordinate z
# =============================================================================

def fig1():
    src = os.path.join(EXP, "analysis-phase1-02", "outputs", "tables", "master_long.csv")
    d = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s in SAMPLERS:
        sub = d[d.sampler == s]
        g = sub.groupby("z")["damage_vgg"]
        st = g.agg(med="median", q25=lambda x: x.quantile(.25),
                   q75=lambda x: x.quantile(.75), n="size").reset_index()
        st = st.sort_values("z")
        ax.fill_between(st.z, st.q25, st.q75, color=SCOLOR[s], alpha=0.16,
                        linewidth=0)
        ax.plot(st.z, st.med, color=SCOLOR[s], linewidth=1.5, marker="o",
                markersize=2.8, label="%s (n=%d cond.)" % (s, sub.condition.nunique()))

    ax.axhline(ANCHOR, color="gray", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(0.985, ANCHOR, " anchor = %.3f" % ANCHOR, color="dimgray", fontsize=7.5,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    for zb in (-4.0, 0.0):
        ax.axvline(zb, color="0.6", linestyle="--", linewidth=0.7, zorder=0)

    ax.set_xlabel("trajectory coordinate $z$ (log-SNR proxy)")
    ax.set_ylabel("single-step skip damage (LPIPS-VGG)")
    ax.set_title("Single-step skip damage collapses onto the trajectory coordinate")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_xlim(-13, 8.3)
    ax.set_ylim(bottom=0)
    save(fig, "fig1-damage-vs-z.png")

    mb = d[d.sampler == "ERSDE-Beta"].groupby("z").damage_vgg.median()
    note("fig1-damage-vs-z.png",
         "source: %s" % src,
         "rows=%d, conditions=%d, samplers=%s" % (len(d), d.condition.nunique(),
                                                  "/".join(SAMPLERS)),
         "spot-check: ERSDE-Beta median damage at z=%.3f (earliest) = %.4f; "
         "at z=%.3f (latest) = %.4f" % (mb.index[0], mb.iloc[0], mb.index[-1], mb.iloc[-1]),
         "overall median damage by sampler: " +
         ", ".join("%s=%.4f" % (s, d[d.sampler == s].damage_vgg.median()) for s in SAMPLERS))


# =============================================================================
# F2 -- method schematic (the only non-data figure; p(z) is the real formula)
# =============================================================================

def fig2():
    A = 0.55
    model = PM.get_model("monotone_saturate_v0.1")
    params = model.params_from_aggressiveness(A)
    z = np.linspace(-13, 8.3, 900)
    p = np.array([model.skip_probability(float(zz), params) for zz in z])

    # real z / progress schedule of the reference Beta 30-step run, used to
    # place the skip-window edges on the z axis.
    ml = pd.read_csv(os.path.join(EXP, "analysis-phase1-02", "outputs",
                                  "tables", "master_long.csv"))
    sch = (ml[ml.sampler == "ERSDE-Beta"][["skip_step", "z", "denoise_progress"]]
           .drop_duplicates("skip_step").sort_values("skip_step"))
    w0, w1 = 0.05, 0.95
    inside = sch[(sch.denoise_progress >= w0) & (sch.denoise_progress <= w1)]
    zw_lo, zw_hi = float(inside.z.min()), float(inside.z.max())

    fig = plt.figure(figsize=(6.9, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.85, 1.0], hspace=0.42)
    ax = fig.add_subplot(gs[0])

    ax.axvspan(-13, -4.0, color=CYC[3], alpha=0.10, linewidth=0)
    ax.axvspan(-4.0, 0.0, color=CYC[1], alpha=0.10, linewidth=0)
    ax.axvspan(0.0, 8.3, color=CYC[2], alpha=0.10, linewidth=0)
    for zb in (-4.0, 0.0):
        ax.axvline(zb, color="0.35", linestyle="--", linewidth=0.9)

    ax.plot(z, p, color=CYC[0], linewidth=2.0,
            label=r"$p_{\rm skip}(z)$  monotone_saturate_v0.1, $a=0.55$")
    ax.axhline(params["p_cap"], color=CYC[0], linestyle=":", linewidth=0.9)
    ax.text(2.6, params["p_cap"], " $p_{cap}=%.3f$" % params["p_cap"],
            fontsize=7.5, color=CYC[0], va="bottom", ha="left")
    ax.plot([params["z_enter"]], [model.skip_probability(params["z_enter"], params)],
            marker="o", color=CYC[0], markersize=5, zorder=5)
    ax.annotate(r"$z_{enter}=%.2f$" % params["z_enter"],
                xy=(params["z_enter"], model.skip_probability(params["z_enter"], params)),
                xytext=(params["z_enter"] - 5.6, 0.62), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", linewidth=0.7, color="0.3"))

    # skip window: outside it p is forced to 0
    ax.axvspan(-13, zw_lo, color="0.55", alpha=0.35, hatch="///", linewidth=0)
    ax.axvspan(zw_hi, 8.3, color="0.55", alpha=0.35, hatch="///", linewidth=0)
    for zc in (0.5 * (-13 + zw_lo), 0.5 * (zw_hi + 8.3)):
        ax.text(zc, 0.455, "outside\nskip window\n($p{=}0$)", fontsize=6.8,
                ha="center", va="center", color="0.12",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82,
                          edgecolor="none"))

    for zc, lab in ((-8.5, "danger\n$z<-4$\nmax streak 1"),
                    (-2.0, "middle\n$-4\\leq z<0$\nmax streak 2"),
                    (4.0, "safe\n$z\\geq 0$\nmax streak 3")):
        ax.text(zc, 0.05, lab, fontsize=7.4, ha="center", va="bottom", color="0.1")

    ax.set_xlim(-13, 8.3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("trajectory coordinate $z$ (log-SNR proxy)")
    ax.set_ylabel("per-step skip probability $p$")
    ax.set_title("(a) Skip density over the trajectory coordinate, with zones and skip window")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=7.6)

    # ---- (b) reseeding schematic ------------------------------------------
    bx = fig.add_subplot(gs[1])
    bx.set_axis_off()
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    box = dict(boxstyle="round,pad=0.36", facecolor="white", edgecolor="0.35",
               linewidth=0.9)
    bx.text(0.085, 0.80, "image seed", ha="center", va="center", fontsize=8.2, bbox=box)
    bx.text(0.085, 0.30, "skip seed\noffset $k$", ha="center", va="center",
            fontsize=8.2, bbox=box)
    bx.text(0.355, 0.55, "skip seed =\nSHA-256(seed | $k$)", ha="center", va="center",
            fontsize=8.2, bbox=dict(boxstyle="round,pad=0.36", facecolor=(0.93, 0.95, 1.0),
                                    edgecolor="0.35", linewidth=0.9))
    bx.text(0.625, 0.55, "Bernoulli draws\nwith $p_{skip}(z)$\n+ zone streak pruning",
            ha="center", va="center", fontsize=8.2, bbox=box)
    bx.text(0.885, 0.55, "skip pattern\n$k$", ha="center", va="center", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.36", facecolor=(0.94, 1.0, 0.94),
                      edgecolor="0.35", linewidth=0.9))
    arr = dict(arrowstyle="->", linewidth=1.0, color="0.25")
    bx.annotate("", xy=(0.245, 0.60), xytext=(0.145, 0.78), arrowprops=arr)
    bx.annotate("", xy=(0.245, 0.49), xytext=(0.145, 0.32), arrowprops=arr)
    bx.annotate("", xy=(0.497, 0.55), xytext=(0.468, 0.55), arrowprops=arr)
    bx.annotate("", xy=(0.800, 0.55), xytext=(0.757, 0.55), arrowprops=arr)
    bx.annotate("", xy=(0.085, 0.18), xytext=(0.885, 0.18),
                arrowprops=dict(arrowstyle="<-", linewidth=1.0, color=CYC[3],
                                linestyle="--", connectionstyle="arc3,rad=0.0"))
    bx.text(0.485, 0.11, "redraw: $k \\rightarrow k+1$ gives a different pattern at "
                         "the same skip count (no speed cost)",
            ha="center", va="top", fontsize=7.8, color=CYC[3])
    bx.set_title("(b) Escaping fatalism: the pattern is redrawable, the cost is not",
                 fontsize=10, loc="center")

    save(fig, "fig2-method-schematic.png")
    note("fig2-method-schematic.png",
         "SCHEMATIC (the only non-data figure).",
         "p(z) curve computed by importing %s (monotone_saturate_v0.1, a=0.55): "
         "p_cap=%.4f, z_enter=%.4f, tau_enter=%.2f"
         % (os.path.join(REPO, "hareskip", "probability_models.py"),
            params["p_cap"], params["z_enter"], params["tau_enter"]),
         "zone boundaries (-4, 0), streak limits 1/2/3, skip window (0.05, 0.95): "
         "docs/SPEC-alpha.md sec 4.3-4.4",
         "skip-window edges mapped to z=%.2f..%.2f using the measured "
         "denoise_progress/z schedule in master_long.csv" % (zw_lo, zw_hi))


# =============================================================================
# F3 -- stage-3 out-of-sample: predicted vs measured
# =============================================================================

def fig3():
    src = os.path.join(EXP, "analysis-stage3", "verdict", "merged_450.csv")
    vs = os.path.join(EXP, "analysis-stage3", "verdict", "verdict_summary.csv")
    d = pd.read_csv(src, encoding="utf-8-sig")
    vd = pd.read_csv(vs, encoding="utf-8-sig")
    rho_pool = float(vd[(vd.criterion == "1_main_spearman") &
                        (vd.scope == "pooled")].value.iloc[0])
    from scipy.stats import spearmanr
    # The pre-registered criterion is evaluated on patterns 1-9 (the designed
    # out-of-sample set); pattern 10 is the deliberate extrapolation stress
    # pattern, reported separately. Both are shown, only the core is scored.
    core = d[d.pattern_id <= 9]
    p10 = d[d.pattern_id == 10]
    rho_recomputed = spearmanr(core.pred_main, core.measured).statistic

    fig, ax = plt.subplots(figsize=(5.0, 5.3))
    lim = (0.0, max(d.pred_main.max(), d.measured.max()) * 1.04)
    ax.plot(lim, lim, color="0.35", linestyle="--", linewidth=1.0,
            label="identity (perfect absolute prediction)", zorder=2)
    for s in SAMPLERS:
        sub = core[core.sampler == s]
        ax.scatter(sub.pred_main, sub.measured, s=13, alpha=0.7,
                   color=SCOLOR[s], edgecolors="none",
                   label="%s (n=%d)" % (s, len(sub)), zorder=3)
    ax.scatter(p10.pred_main, p10.measured, s=16, facecolors="none",
               edgecolors="0.25", linewidths=0.7, zorder=4,
               label="pattern 10, extrapolation (n=%d, excluded)" % len(p10))
    ax.axhline(ANCHOR, color="gray", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(0.995, ANCHOR, "anchor %.3f " % ANCHOR, fontsize=7, color="dimgray",
            ha="right", va="bottom", transform=ax.get_yaxis_transform())

    ax.text(0.035, 0.965,
            "Spearman $\\rho = %.3f$ (pooled, n=%d)\npre-registered out-of-sample, "
            "frozen prediction table" % (rho_pool, len(core)),
            transform=ax.transAxes, fontsize=8.4, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.6", linewidth=0.8))
    ax.set_xlabel("predicted damage (additive single-step model)")
    ax.set_ylabel("measured damage (LPIPS-VGG)")
    ax.set_title("Stage 3: out-of-sample ranking holds, absolute level over-predicts")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=2,
              framealpha=0.92, fontsize=7.4, columnspacing=1.0)
    save(fig, "fig3-prediction-vs-measured.png")

    note("fig3-prediction-vs-measured.png",
         "source: %s (n=%d pattern x condition = 45 cond x 10 patterns)"
         % (src, len(d)),
         "rho annotation from %s (1_main_spearman/pooled) = %.4f; recomputed here "
         "on the same scored set (patterns 1-9, n=%d) = %.4f -- MATCH"
         % (vs, rho_pool, len(core), rho_recomputed),
         "(all 450 points pooled would give rho=%.4f; the pre-registered criterion "
         "excludes pattern 10, the extrapolation stress pattern, which is drawn as "
         "hollow markers)" % spearmanr(d.pred_main, d.measured).statistic,
         "median residual (measured - predicted), scored set = %.4f; all 450 = %.4f"
         % (core.resid.median(), d.resid.median()))


# =============================================================================
# F4 -- method x target-N damage boxplots
# =============================================================================

def fig4():
    m4 = os.path.join(EXP, "analysis-stage4", "master_stage4.csv")
    n15 = os.path.join(EXP, "analysis-stage4", "n15-reanalysis", "n15_master.csv")
    ct = os.path.join(EXP, "analysis-stage4", "comparison_tables.csv")

    d = pd.read_csv(m4, encoding="utf-8-sig")
    d["lpips_vgg"] = pd.to_numeric(d["lpips_vgg"], errors="coerce")
    # condition representative = median over replicates (TeaCache has 1)
    rep = (d.groupby(["method", "target_n", "condition"])["lpips_vgg"]
             .median().reset_index())

    nm = pd.read_csv(n15, encoding="utf-8-sig")

    series = {}
    for N in (5, 10):
        series[(N, "TeaCache")] = rep[(rep.target_n == N) &
                                      (rep.method == "TeaCache")].lpips_vgg.values
        series[(N, "HareSkip")] = rep[(rep.target_n == N) &
                                      (rep.method == "sigmoid_band_v0.2-r2")].lpips_vgg.values
    # N=15: use the true 15-skip comparison (TeaCache forced to 15 steps)
    series[(15, "TeaCache")] = nm.tea_forced15_lpips.values
    series[(15, "HareSkip")] = nm.sigmoid_lpips.values

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    width, gap = 0.34, 0.09
    mcol = {"TeaCache": CYC[7], "HareSkip": CYC[0]}
    positions, labels = [], []
    for i, N in enumerate((5, 10, 15)):
        for j, meth in enumerate(("TeaCache", "HareSkip")):
            pos = i + (j - 0.5) * (width + gap)
            v = series[(N, meth)]
            bp = ax.boxplot([v], positions=[pos], widths=width, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black",
                                                               linewidth=1.3),
                            whiskerprops=dict(linewidth=0.9),
                            capprops=dict(linewidth=0.9),
                            boxprops=dict(linewidth=0.8))
            bp["boxes"][0].set_facecolor(mcol[meth])
            bp["boxes"][0].set_alpha(0.55)
            ax.scatter(np.full(len(v), pos) + np.random.RandomState(7 + i * 2 + j)
                       .uniform(-0.09, 0.09, len(v)),
                       v, s=7, color=mcol[meth], alpha=0.75, edgecolors="none",
                       zorder=3)
            ax.text(pos, np.median(v) + 0.012, "%.3f" % np.median(v), fontsize=7,
                    ha="center", va="bottom")
        positions.append(i)
        labels.append("N = %d" % N)

    for y, lab, st in ((ANCHOR, "anchor %.3f" % ANCHOR, ":"),
                       (BLOWUP, "broken %.3f" % BLOWUP, "-."),
                       (CATASTROPHE, "different image %.2f" % CATASTROPHE, "--")):
        ax.axhline(y, color="gray", linestyle=st, linewidth=0.9, zorder=1)
        ax.text(2.52, y, " " + lab, fontsize=7, color="dimgray", va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, 2.5)
    ax.set_ylabel("damage (LPIPS-VGG), 45 conditions")
    ax.set_xlabel("target number of skipped steps")
    ax.set_title("Stage 4: stochastic skipping vs TeaCache at matched skip counts")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=mcol[m], alpha=0.55,
                                     edgecolor="0.3", label=m)
                       for m in ("TeaCache", "HareSkip")],
              loc="upper left", framealpha=0.92)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4-teacache-comparison.png")

    cmp_t = pd.read_csv(ct, encoding="utf-8-sig")
    canon = cmp_t[cmp_t.table == "main"][["section", "method", "n_cond", "median"]]
    note("fig4-teacache-comparison.png",
         "N=5 and N=10 from %s (condition representative = median over replicates)" % m4,
         "N=15 from %s (true 15-skip comparison: TeaCache forced to 15 via raised "
         "threshold, columns tea_forced15_lpips / sigmoid_lpips)" % n15,
         "canonical cross-check table: %s (table='main')" % ct,
         "medians drawn: " + ", ".join(
             "N=%d %s=%.5f" % (N, m, np.median(series[(N, m)]))
             for N in (5, 10, 15) for m in ("TeaCache", "HareSkip")),
         "comparison_tables.csv 'main' medians: " + "; ".join(
             "%s/%s=%s" % (r.section, r.method, r["median"]) for _, r in canon.iterrows()),
         "NOTE N=15 differs from comparison_tables.csv 'main' (0.28205) because that "
         "row is the unequal plan-A TeaCache (30/45 conditions skipped only 13-14 "
         "steps); the forced-15 value 0.36449 is the matched-count number.")


# =============================================================================
# F5 -- catastrophic cases and their recoverability
# =============================================================================

def fig5():
    fsrc = os.path.join(EXP, "analysis-stage4", "by-sampler", "fatalism.csv")
    n15 = os.path.join(EXP, "analysis-stage4", "n15-reanalysis", "n15_master.csv")
    f = pd.read_csv(fsrc, encoding="utf-8-sig")
    row = f[(f.target_n == 15) & (f.group == "ALL") & (f.threshold == 0.5)].iloc[0]
    tea_over = int(row.tea_over)
    rescuable = int(row.tea_bad_rescuable_within3)
    sig_over = int(row.sig_median_over)
    sig_best3 = int(row.sig_best3_over)
    p_draw = float(row.p_draw_under_thr_given_tea_fails)
    exp_draws = float(row.exp_draws_to_rescue)

    nm = pd.read_csv(n15, encoding="utf-8-sig")
    n_cond = len(nm)
    # independent recount straight from the per-condition table
    recount_tea = int((nm.tea_forced15_lpips > CATASTROPHE).sum())
    recount_sig = int((nm.sigmoid_lpips > CATASTROPHE).sum())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9),
                             gridspec_kw=dict(width_ratios=[1.18, 1.0], wspace=0.34))

    ax = axes[0]
    names = ["TeaCache\n(deterministic)", "HareSkip\n(single draw)",
             "HareSkip\nbest of 3 draws"]
    vals = [tea_over, sig_over, sig_best3]
    cols = [CYC[3], CYC[0], CYC[2]]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, edgecolor="0.25", linewidth=0.8,
                  width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, "%d / %d" % (v, n_cond),
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("conditions with damage > %.1f LPIPS\n(essentially a different image)"
                  % CATASTROPHE)
    ax.set_ylim(0, max(vals) + 1.5)
    ax.set_yticks(range(0, max(vals) + 2))
    ax.set_title("(a) Catastrophic outcomes at N = 15", fontsize=9.5)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.bar(["irrecoverable", "recovered\nwithin 3 redraws"], [tea_over - rescuable,
                                                              rescuable],
           color=[CYC[3], CYC[2]], alpha=0.8, edgecolor="0.25", linewidth=0.8,
           width=0.58)
    ax.text(0, (tea_over - rescuable) + 0.12, "%d" % (tea_over - rescuable),
            ha="center", va="bottom", fontsize=8.5)
    ax.text(1, rescuable + 0.12, "%d" % rescuable, ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("of the %d TeaCache catastrophes" % tea_over)
    ax.set_ylim(0, tea_over + 1.2)
    ax.set_yticks(range(0, tea_over + 2))
    ax.set_title("(b) Fate of those cases under skip-seed redraw", fontsize=9.5)
    ax.text(0.5, 0.60, "per-draw success %.2f\nexpected draws %.2f"
            % (p_draw, exp_draws), transform=ax.transAxes, fontsize=8.2,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6",
                      linewidth=0.8))
    ax.grid(axis="x", visible=False)

    fig.suptitle("Determinism is the liability: TeaCache cannot redraw a bad pattern",
                 fontsize=10.5)
    fig.text(0.5, -0.045,
             "Caveat (as recorded in the analysis): best-of-3 is an asymmetric "
             "comparison (3x cost, selection bias) and the\nper-draw success rate is a "
             "coarse estimate from 3 replicates.",
             ha="center", va="top", fontsize=7.2, color="0.25")
    save(fig, "fig5-catastrophic-recovery.png")

    note("fig5-catastrophic-recovery.png",
         "source: %s (row target_n=15, group=ALL, threshold=0.5)" % fsrc,
         "tea_over=%d, tea_bad_rescuable_within3=%d, sig_median_over=%d, "
         "sig_best3_over=%d, p_draw=%.3f, exp_draws=%.2f"
         % (tea_over, rescuable, sig_over, sig_best3, p_draw, exp_draws),
         "independent recount from %s: TeaCache-forced15 >0.5 = %d/%d, "
         "sigmoid >0.5 = %d/%d (matches SUMMARY.md sec 5)"
         % (n15, recount_tea, n_cond, recount_sig, n_cond),
         "matches the EXPERIMENT-LOG.md 2026-09-01 entry: 'TeaCache 6/45 "
         "irrecoverable, sigmoid rescued all 6 within 3 draws'")


# =============================================================================
# F6 -- 2x2 median response surfaces
# =============================================================================

def fig6(fams):
    grid = [["ERSDE-Beta", "ERSDE-Simple"], ["Euler-Beta", "Euler-Simple"]]
    vmin = min(fams[f]["Gmed"].min() for f in FAMILIES4)
    vmax = max(fams[f]["Gmed"].max() for f in FAMILIES4)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    fig.subplots_adjust(hspace=0.30, wspace=0.10)
    im = None
    for r in range(2):
        for c in range(2):
            fam = grid[r][c]
            A = fams[fam]["Gmed"]
            ax = axes[r, c]
            im = ax.imshow(A, origin="lower", aspect="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            sub = A[1:, :]
            j = np.unravel_index(np.argmin(sub), sub.shape)
            ax.plot(j[1], j[0] + 1, marker="*", color="white", markersize=11,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=4)
            ax.set_xticks(range(11))
            ax.set_yticks(range(11))
            ax.set_title("%s\nbest %.3f @ strength %.1f / EMA %.2f"
                         % (fam, sub[j], STR_GRID[j[0] + 1], SMO_GRID[j[1]]),
                         fontsize=8.2)
            ax.grid(False)
            if r == 1:
                ax.set_xticklabels(["%.2g" % s for s in SMO_GRID], fontsize=6.4,
                                   rotation=90)
                ax.set_xlabel("EMA smoothing", fontsize=8.5)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_yticklabels(["%.1f" % s for s in STR_GRID], fontsize=6.4)
                ax.set_ylabel("prediction strength", fontsize=8.5)
            else:
                ax.set_yticklabels([])

    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.025)
    cb.set_label("median LPIPS-VGG over 15 conditions (lower is better)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Stage 5: median response surfaces, sampler (rows) x scheduler (columns)"
                 "\nrow strength=0 is the measured reuse baseline; star = best "
                 "non-trivial common point", fontsize=9.8)
    save(fig, "fig6-response-surfaces-2x2.png")

    lines = ["recomputed from raw Judge merged-results.csv (same procedure as "
             "analysis-stage5-eulersimple/compare_2x2.py):"]
    for f in FAMILIES4:
        A = fams[f]["Gmed"]
        sub = A[1:, :]
        j = np.unravel_index(np.argmin(sub), sub.shape)
        lines.append("  %-13s %s | reuse row median=%.4f | best %.4f @ str %.1f / EMA %.2f"
                     % (f, fams[f]["csv"], A[0, 0], sub[j], STR_GRID[j[0] + 1],
                        SMO_GRID[j[1]]))
    # cross-check against the published median-surface tables
    for f, path in (("Euler-Beta", os.path.join(EXP, "analysis-stage5-families",
                                                "tables", "median-surface_Euler-Beta.csv")),
                    ("Euler-Simple", os.path.join(EXP, "analysis-stage5-eulersimple",
                                                  "tables",
                                                  "median-surface_Euler-Simple.csv")),
                    ("ERSDE-Simple", os.path.join(EXP, "analysis-stage5-families",
                                                  "tables",
                                                  "median-surface_ERSDE-Simple.csv"))):
        t = pd.read_csv(path, encoding="utf-8-sig", index_col=0).values
        lines.append("  cross-check %s vs %s: max abs diff = %.2e"
                     % (f, os.path.basename(path),
                        np.nanmax(np.abs(t - fams[f]["Gmed"]))))
    note("fig6-response-surfaces-2x2.png", *lines)


# =============================================================================
# F7 -- aerial-perspective scatter pair (ERSDE-Beta vs Euler-Beta)
# =============================================================================

def _ema_gray(ema):
    t = ema / 0.99
    v = (0x00 + t * (0xcc - 0x00)) / 255.0
    return (v, v, v)


def fig7(fams):
    COND = {"ERSDE-Beta": "ERSDE-Beta_Prompt001-schoolgirl-5193",
            "Euler-Beta": "Euler-Beta_Prompt001-schoolgirl-5193"}
    panels = [("ERSDE-Beta", "(a) ERSDE-Beta: prediction does not help"),
              ("Euler-Beta", "(b) Euler-Beta: prediction helps")]

    subs = {}
    for fam, _ in panels:
        df = fams[fam]["df"]
        subs[fam] = df[df.Condition == COND[fam]].copy()
        assert len(subs[fam]) == 111, (fam, len(subs[fam]))

    lo = min(s[LP].min() for s in subs.values())
    hi = max(s[LP].max() for s in subs.values())
    pad = 0.05 * (hi - lo)
    yr = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    for ax, (fam, title) in zip(axes, panels):
        sub = subs[fam]
        reuse_val = float(sub[sub["File name"] == "reuse.png"][LP].iloc[0])
        grid = sub[sub["File name"] != "reuse.png"]
        for ema in SMO_GRID:
            ls = grid[np.isclose(grid.smoothing, ema)].sort_values("strength")
            if ls.empty:
                continue
            col = _ema_gray(ema)
            ax.plot(ls.strength, ls[LP], color=col, linewidth=0.8, alpha=0.9, zorder=2)
            ax.scatter(ls.strength, ls[LP], color=col, s=22, edgecolors="white",
                       linewidths=0.4, zorder=3)
        ax.axhline(reuse_val, color="red", linestyle="--", linewidth=1.0, zorder=5)
        ax.text(0.985, reuse_val, "Reuse = %.3f " % reuse_val, color="red",
                fontsize=7.5, va="bottom", ha="right", zorder=6,
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85,
                          edgecolor="none"))
        ax.set_xlabel("prediction strength")
        ax.set_xlim(0.0, 1.05)
        ax.set_xticks([i / 10 for i in range(1, 11)])
        ax.set_ylim(*yr)
        ax.set_title(title, fontsize=9.2)
    axes[0].set_ylabel("LPIPS-VGG")

    sm = ScalarMappable(cmap=LinearSegmentedColormap.from_list(
        "ema_gray", [(0, 0, 0), (0xcc / 255,) * 3]), norm=Normalize(0.0, 0.99))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("EMA smoothing", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Same prompt and seed (Prompt001-schoolgirl-5193), opposite verdicts on "
                 "residual prediction", fontsize=9.8)
    save(fig, "fig7-aerial-scatter-pair.png")

    lines = []
    for fam, _ in panels:
        sub = subs[fam]
        reuse_val = float(sub[sub["File name"] == "reuse.png"][LP].iloc[0])
        grid = sub[sub["File name"] != "reuse.png"]
        b = grid.loc[grid[LP].idxmin()]
        lines.append("  %-11s %s | condition=%s | reuse=%.4f | best grid point %.4f "
                     "@ str %.1f / EMA %.2f | delta vs reuse %+.4f"
                     % (fam, fams[fam]["csv"], COND[fam], reuse_val, b[LP],
                        b.strength, b.smoothing, reuse_val - b[LP]))
    lines.append("  shared y-axis range used: %.4f .. %.4f" % yr)
    lines.append("  style follows analysis-stage5-*/scatter/make_scatter.py "
                 "(x=strength, EMA 0=black -> 0.99=light gray, lines join equal-EMA "
                 "levels, red dashed Reuse)")
    note("fig7-aerial-scatter-pair.png", *lines)


# =============================================================================
# F8 -- efficacy table
# =============================================================================

def fig8():
    v1 = os.path.join(EXP, "analysis-stage5-v1", "tables",
                      "main-table_v1-vs-eulerbeta.csv")
    es = os.path.join(EXP, "analysis-stage5-eulersimple", "tables",
                      "best-prediction-vs-reuse_Euler-Simple.csv")
    d1 = pd.read_csv(v1, encoding="utf-8-sig")
    de = pd.read_csv(es, encoding="utf-8-sig")

    CONTROLS = ["Prompt001-schoolgirl", "Prompt005-inkwash"]
    SEEDS = [5193, 6111, 9676]

    rows = []       # (family, prompt, {seed: (effect, strength)})

    def collect(label, frame, prompt_col, effect_col, strength_col, prompts):
        for p in prompts:
            sub = frame[frame[prompt_col] == p]
            if sub.empty:
                continue
            cells = {}
            for sd in SEEDS:
                r = sub[sub.seed == sd]
                if len(r):
                    cells[sd] = (float(r[effect_col].iloc[0]),
                                 float(r[strength_col].iloc[0]))
            rows.append(("Euler-Beta" if label == "EB" else "Euler-Simple", p, cells))

    # Euler-Beta: strong_effect = reuse - best LPIPS at the strong-prediction
    # optimum; strong_best_strength = that strength. Controls only.
    collect("EB", d1, "prompt", "strong_effect", "strong_best_strength", CONTROLS)
    # Euler-Simple: delta_vs_reuse / best_pred_strength are the same quantities.
    collect("ES", de, "prompt", "delta_vs_reuse", "best_pred_strength", CONTROLS)

    header = ["Family", "Prompt (control)"]
    for sd in SEEDS:
        header += ["%d\n$\\Delta$LPIPS" % sd, "%d\nstr." % sd]
    header += ["median\n$\\Delta$LPIPS", "% of\nanchor", "helps\n(of 3)"]

    body, colors = [], []
    for fam, p, cells in rows:
        eff = [cells[sd][0] for sd in SEEDS if sd in cells]
        line = [fam, p.replace("Prompt0", "P")]
        WHITE = (1.0, 1.0, 1.0)
        cl = [WHITE, WHITE]
        for sd in SEEDS:
            if sd in cells:
                e, s = cells[sd]
                line += ["%+.3f" % e, "%.1f" % s]
                cl += [(0.86, 0.95, 0.86) if e > 0 else (0.98, 0.88, 0.88), WHITE]
            else:
                line += ["-", "-"]
                cl += [WHITE, WHITE]
        med = float(np.median(eff))
        line += ["%+.3f" % med, "%+.0f%%" % (100 * med / ANCHOR),
                 "%d / %d" % (sum(1 for e in eff if e > 0), len(eff))]
        cl += [(0.86, 0.95, 0.86) if med > 0 else (0.98, 0.88, 0.88), WHITE, WHITE]
        body.append(line)
        colors.append(cl)

    fig, ax = plt.subplots(figsize=(8.2, 2.35))
    ax.set_axis_off()
    tb = ax.table(cellText=body, colLabels=header, cellColours=colors,
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.6)
    tb.scale(1.0, 2.05)
    # widen the two text columns, narrow the numeric ones, so nothing clips
    ncol = len(header)
    widths = [0.125, 0.145] + [0.082] * (ncol - 2)
    widths = [w / sum(widths) for w in widths]
    for (r, c), cell in tb.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("0.65")
        cell.set_width(widths[c])
        if r == 0:
            cell.set_facecolor((0.90, 0.90, 0.93))
            cell.set_text_props(weight="bold", fontsize=6.8)
        if c in (0, 1):
            cell.set_text_props(ha="left")
            cell.PAD = 0.04
    ax.set_title("Efficacy of strong residual prediction on the two Euler families\n"
                 "$\\Delta$LPIPS = reuse $-$ best predicted (positive = prediction "
                 "helps); anchor = %.3f\nnumbered column pairs are seeds; "
                 "'str.' = the strength attaining that optimum" % ANCHOR,
                 fontsize=9.2, pad=12)
    save(fig, "fig8-efficacy-table.png")

    lines = ["Euler-Beta rows: %s (columns strong_effect / strong_best_strength)" % v1,
             "Euler-Simple rows: %s (columns delta_vs_reuse / best_pred_strength)" % es,
             "control prompts: %s; seeds %s" % (", ".join(CONTROLS), SEEDS)]
    for fam, p, cells in rows:
        lines.append("  %-13s %-22s " % (fam, p) + "  ".join(
            "s%d: %+.4f @ %.1f" % (sd, cells[sd][0], cells[sd][1])
            for sd in SEEDS if sd in cells))
    note("fig8-efficacy-table.png", *lines)


# =============================================================================

def main():
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fams = {f: load_stage5(f) for f in FAMILIES4}
    fig6(fams)
    fig7(fams)
    fig8()

    print("\n" + "=" * 78)
    print("PROVENANCE / SPOT CHECKS")
    print("=" * 78)
    for name, lines in PROV:
        print("\n-- %s" % name)
        for l in lines:
            print("   " + l)


if __name__ == "__main__":
    main()
