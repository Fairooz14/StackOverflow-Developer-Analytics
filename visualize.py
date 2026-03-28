import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_DIR = "charts"

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
PANEL   = "#F8F9FA"
BORDER  = "#DEE2E6"
TEXT    = "#212529"
MUTED   = "#6C757D"
ACCENT1 = "#2563EB"   # blue  – primary
ACCENT2 = "#16A34A"   # green – secondary
ACCENT3 = "#DC2626"   # red   – alert
ACCENT4 = "#7C3AED"   # purple
ACCENT5 = "#D97706"   # amber
GOLD    = "#CA8A04"
SILVER  = "#6B7280"
BRONZE  = "#92400E"

QUAL = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
        "#0891B2", "#BE185D", "#065F46", "#1E40AF", "#78350F"]

# ── Global rcParams ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.color":        BORDER,
    "grid.linewidth":    0.7,
    "grid.alpha":        1.0,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  BG,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": TEXT,
    "legend.fontsize":   9,
    "text.color":        TEXT,
    "font.family":       "DejaVu Sans",
    "lines.linewidth":   2,
    "figure.dpi":        150,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def _k(v):
    if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
    if v >= 1_000:     return f"{v/1_000:.1f}k"
    return str(int(v))

def _save(filename, fig):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"   {filename}")

def _subtitle(ax, text):
    ax.set_title(text, fontsize=9, color=MUTED, pad=4,
                 fontweight="normal", loc="left")

def _spine_clean(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Q01  Tag Momentum — horizontal bar colour-mapped by momentum score
# ─────────────────────────────────────────────────────────────────────────────
def chart_tag_momentum():
    df = pd.read_csv("results/Q01_tag_momentum.csv").head(20)
    df = df.sort_values("momentum_score", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle("Tag Momentum Index", fontsize=15, fontweight="bold",
                 color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Weighted score: 60% recent-volume share + 40% YoY acceleration  (2018-2023)")

    norm   = plt.Normalize(df["momentum_score"].min(), df["momentum_score"].max())
    cmap   = LinearSegmentedColormap.from_list("m", ["#BFDBFE", ACCENT1, "#1E3A8A"])
    colors = [cmap(norm(v)) for v in df["momentum_score"]]

    bars = ax.barh(df["tag"], df["momentum_score"], color=colors,
                   height=0.65, zorder=3, edgecolor=BG, linewidth=0.4)
    ax.set_xlabel("Momentum score", color=MUTED)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _spine_clean(ax)

    for bar, val in zip(bars, df["momentum_score"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=TEXT)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q01_tag_momentum.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q02  Response-Time Percentiles — whisker plot from pre-computed percentiles
# ─────────────────────────────────────────────────────────────────────────────
def chart_response_percentiles():
    df = pd.read_csv("results/Q02_response_time_percentiles.csv")
    df = df.sort_values("p50_hrs").head(20)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Community Response Time by Tag", fontsize=15,
                 fontweight="bold", color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Hours to first accepted answer  —  P25 / P50 / P75 / P90 / P99")

    y = np.arange(len(df))
    ax.barh(y, df["p75_hrs"] - df["p25_hrs"], left=df["p25_hrs"],
            height=0.45, color=ACCENT1, alpha=0.25, zorder=3)
    ax.scatter(df["p50_hrs"], y, color=ACCENT1, s=70, zorder=5)

    for i, row in enumerate(df.itertuples()):
        ax.plot([row.p25_hrs, row.p75_hrs], [i, i], color=ACCENT1, lw=1.8, zorder=4)
        ax.plot([row.p75_hrs, row.p90_hrs], [i, i], color=MUTED, lw=1.2,
                linestyle="--", zorder=4)
        ax.plot([row.p90_hrs, row.p99_hrs], [i, i], color=ACCENT3, lw=0.9,
                linestyle=":", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(df["tag"], fontsize=9, color=TEXT)
    ax.set_xlabel("Hours to first accepted answer", color=MUTED)
    _spine_clean(ax)

    handles = [
        mpatches.Patch(color=ACCENT1, alpha=0.3, label="IQR (P25-P75)"),
        plt.Line2D([], [], color=ACCENT1, marker="o", ms=6, label="P50 median"),
        plt.Line2D([], [], color=MUTED,   ls="--",   label="P75-P90"),
        plt.Line2D([], [], color=ACCENT3, ls=":",    label="P90-P99"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q02_response_percentiles.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q03  Monthly Volume + 3-month Rolling Average
# ─────────────────────────────────────────────────────────────────────────────
def chart_monthly_rolling():
    df = pd.read_csv("results/Q03_monthly_rolling_avg.csv")
    df["period"] = (df["yr"].astype(str) + "-"
                    + df["mo"].astype(str).str.zfill(2))
    df = df[df["yr"] >= 2015].reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle("Monthly Question Volume  (2015-2023)", fontsize=15,
                 fontweight="bold", color=TEXT, x=0.05, ha="left")
    _subtitle(ax1, "Raw count (shaded area) vs 3-month rolling average (line)")

    x = np.arange(len(df))
    ax1.fill_between(x, df["questions"], alpha=0.12, color=ACCENT1, zorder=2)
    ax1.plot(x, df["questions"], color=ACCENT1, linewidth=1.0,
             alpha=0.4, zorder=3)
    ax1.plot(x, df["rolling_3m_questions"], color=ACCENT1, linewidth=2.2,
             zorder=4, label="3-month rolling avg")
    ax1.set_ylabel("Questions posted", color=ACCENT1)
    ax1.tick_params(axis="y", colors=ACCENT1)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _k(v)))

    ax2 = ax1.twinx()
    ax2.plot(x, df["rolling_3m_score"], color=ACCENT5, linewidth=1.5,
             linestyle="--", zorder=3, label="Rolling avg score")
    ax2.set_ylabel("Avg question score", color=ACCENT5)
    ax2.tick_params(axis="y", colors=ACCENT5)
    ax2.set_facecolor("none")

    year_ticks = df[df["mo"] == 1].index.tolist()
    ax1.set_xticks(year_ticks)
    ax1.set_xticklabels(df.loc[year_ticks, "yr"])
    ax1.set_xlim(0, len(df) - 1)
    _spine_clean(ax1)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q03_monthly_rolling.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q04  Engagement Funnel — stacked 100% horizontal bar
# ─────────────────────────────────────────────────────────────────────────────
def chart_engagement_funnel():
    df = pd.read_csv("results/Q04_engagement_funnel.csv")
    df = df.sort_values("total_q", ascending=False).head(14)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Engagement Funnel per Technology", fontsize=15,
                 fontweight="bold", color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Share of questions: unanswered  ->  answered  ->  accepted")

    total = df["total_q"].values
    ans   = df["answered"].values
    acc   = df["accepted"].values

    pct_unanswered = (total - ans) / total * 100
    pct_answered   = (ans - acc)   / total * 100
    pct_accepted   = acc            / total * 100

    y = np.arange(len(df))
    h = 0.6
    ax.barh(y, pct_unanswered, height=h,
            color=ACCENT3, alpha=0.75, label="Unanswered", zorder=3)
    ax.barh(y, pct_answered, left=pct_unanswered, height=h,
            color=ACCENT1, alpha=0.75, label="Answered (no accept)", zorder=3)
    ax.barh(y, pct_accepted, left=pct_unanswered + pct_answered, height=h,
            color=ACCENT2, alpha=0.85, label="Accepted", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["tag"], fontsize=9, color=TEXT)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of questions (%)", color=MUTED)
    ax.axvline(50, color=BORDER, linewidth=1, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    _spine_clean(ax)

    # Annotate accepted-of-answered % using the correct column name
    for i, row in enumerate(df.itertuples()):
        ax.text(101.5, i,
                f"{row.accept_of_answered_pct:.0f}%\nof answered",
                va="center", fontsize=7, color=ACCENT2, linespacing=1.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q04_engagement_funnel.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q06  YoY Rank — bump chart (one line per tag)
# ─────────────────────────────────────────────────────────────────────────────
def chart_yoy_bump():
    df = pd.read_csv("results/Q06_yoy_rank_change.csv")
    years = sorted(df["yr"].unique())
    tags  = (df.groupby("tag")["question_count"].sum()
               .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Tag Rank Movement  (2019-2023)", fontsize=15,
                 fontweight="bold", color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Bump chart — lower rank number = more questions that year")

    color_map = {t: QUAL[i % len(QUAL)] for i, t in enumerate(tags)}

    for tag in tags:
        sub = df[df["tag"] == tag].sort_values("yr")
        if len(sub) < 2:
            continue
        col = color_map[tag]
        ax.plot(sub["yr"], sub["yr_rank"], color=col, linewidth=2,
                marker="o", markersize=7, zorder=3,
                path_effects=[pe.withStroke(linewidth=3.5, foreground=BG)])
        last = sub.iloc[-1]
        ax.text(last["yr"] + 0.1, last["yr_rank"], tag,
                va="center", fontsize=8, color=col, fontweight="bold")

    ax.invert_yaxis()
    ax.set_xticks(years)
    ax.set_xlabel("Year", color=MUTED)
    ax.set_ylabel("Rank  (1 = most questions)", color=MUTED)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _spine_clean(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q06_yoy_bump.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q08  Activity Heatmap — day x hour tile
# ─────────────────────────────────────────────────────────────────────────────
def chart_activity_heatmap():
    df = pd.read_csv("results/Q08_activity_heatmap.csv")
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]
    pivot = (df.pivot_table(index="DayOfWeek", columns="hour_of_day",
                            values="question_count", aggfunc="sum")
               .reindex(days_order))

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle("Posting Activity Heatmap", fontsize=15, fontweight="bold",
                 color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Question count by day of week x hour of day")

    cmap = LinearSegmentedColormap.from_list(
        "act", [PANEL, "#DBEAFE", "#93C5FD", ACCENT1, "#1E3A8A"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, zorder=3)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)],
                       rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(7))
    ax.set_yticklabels(days_order, fontsize=9, color=TEXT)
    ax.set_xlabel("Hour of Day", color=MUTED)
    ax.tick_params(length=0)

    mean_val = np.nanmean(pivot.values)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if pd.notna(val) and val > 0:
                txt_col = BG if val > mean_val else MUTED
                ax.text(c, r, _k(val), ha="center", va="center",
                        fontsize=6.5, color=txt_col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.set_label("Questions", color=MUTED, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q08_activity_heatmap.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q09  Cohort Retention — one curve per join-year
# ─────────────────────────────────────────────────────────────────────────────
def chart_cohort_retention():
    df = pd.read_csv("results/Q09_cohort_retention.csv")
    cohorts = sorted(df["cohort_year"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("User Cohort Retention", fontsize=15, fontweight="bold",
                 color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "% of users from each join-year cohort still posting N years later")

    cmap_r = mpl.colormaps.get_cmap("Blues")
    for i, cohort in enumerate(cohorts):
        sub = df[df["cohort_year"] == cohort].sort_values("years_since_join")
        col = cmap_r(0.35 + 0.55 * i / max(len(cohorts) - 1, 1))
        ax.plot(sub["years_since_join"], sub["retention_pct"],
                marker="o", linewidth=2, markersize=5,
                color=col, label=str(cohort), zorder=3)

    ax.set_xlabel("Years since joining", color=MUTED)
    ax.set_ylabel("Retention (%)", color=MUTED)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(0, 108)
    ax.axhline(50, color=BORDER, linestyle="--", linewidth=1.0)
    ax.text(5.08, 50, "50%", va="center", fontsize=8, color=MUTED)
    ax.legend(title="Join year", ncol=2, title_fontsize=8, fontsize=8)
    _spine_clean(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q09_cohort_retention.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q10  Answer Quality by User Tier — one grouped bar chart per tag
# ─────────────────────────────────────────────────────────────────────────────
def chart_answer_quality_tiers():
    df = pd.read_csv("results/Q10_answer_quality_by_tier_tag.csv")
    tags  = df["tag"].unique()[:8]
    tiers = ["beginner", "intermediate", "expert"]
    tier_colors = [ACCENT3, ACCENT1, ACCENT2]
    x = np.arange(len(tiers))

    for tag in tags:
        sub = df[df["tag"] == tag]
        vals = [sub[sub["Tier"] == t]["avg_score"].values[0]
                if t in sub["Tier"].values else 0
                for t in tiers]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"Answer Quality  —  {tag.upper()}", fontsize=14,
                     fontweight="bold", color=TEXT, x=0.05, ha="left")
        _subtitle(ax, "Avg answer score by user reputation tier")

        bars = ax.bar(x, vals, color=tier_colors, width=0.5, zorder=3,
                      edgecolor=BG, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tiers],
                           fontsize=10, color=TEXT)
        ax.set_ylabel("Avg answer score", color=MUTED)
        _spine_clean(ax)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", fontsize=9,
                    color=TEXT, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe_tag = tag.replace("-", "_").replace(".", "")
        _save(f"Q10_answer_quality_{safe_tag}.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q11  Badge Velocity — two separate charts
# ─────────────────────────────────────────────────────────────────────────────
def chart_badge_velocity():
    df = pd.read_csv("results/Q11_badge_velocity.csv")
    tiers = df["Tier"].tolist()
    tier_colors = {"beginner": ACCENT3, "intermediate": ACCENT1, "expert": ACCENT2}

    # A — days to first badge lollipop
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Days to First Badge by User Tier", fontsize=14,
                 fontweight="bold", color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Median number of days from account creation to earning first badge")

    medians = df["median_days"].tolist()
    cols    = [tier_colors.get(t, MUTED) for t in tiers]
    y       = np.arange(len(tiers))
    ax.hlines(y, 0, medians, colors=BORDER, linewidth=2)
    ax.scatter(medians, y, color=cols, s=140, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([t.capitalize() for t in tiers], fontsize=10, color=TEXT)
    ax.set_xlabel("Median days to first badge", color=MUTED)
    _spine_clean(ax)
    for i, (val, col) in enumerate(zip(medians, cols)):
        ax.text(val + 0.5, i, f"{int(val)} days",
                va="center", fontsize=9, color=col, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q11a_badge_days.png", fig)

    # B — badge mix stacked bar
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.suptitle("Badge Mix by User Tier", fontsize=14,
                  fontweight="bold", color=TEXT, x=0.05, ha="left")
    _subtitle(ax2, "Average count of bronze / silver / gold badges earned")

    x = np.arange(len(tiers))
    ax2.bar(x, df["avg_bronze"], 0.5, color=BRONZE, label="Bronze", zorder=3)
    ax2.bar(x, df["avg_silver"], 0.5, bottom=df["avg_bronze"],
            color=SILVER, label="Silver", zorder=3)
    ax2.bar(x, df["avg_gold"], 0.5,
            bottom=df["avg_bronze"] + df["avg_silver"],
            color=GOLD, label="Gold", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.capitalize() for t in tiers], fontsize=10, color=TEXT)
    ax2.set_ylabel("Avg badge count", color=MUTED)
    ax2.legend(fontsize=9)
    _spine_clean(ax2)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q11b_badge_mix.png", fig2)


# ─────────────────────────────────────────────────────────────────────────────
# Q12  Geographic Productivity Index
# ─────────────────────────────────────────────────────────────────────────────
def chart_geo_productivity():
    df = pd.read_csv("results/Q12_geo_productivity.csv").head(15)
    df = df.sort_values("productivity_index", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Geographic Productivity Index", fontsize=15,
                 fontweight="bold", color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Total question score / user count per country  (min 50 users)")

    norm   = plt.Normalize(df["productivity_index"].min(),
                           df["productivity_index"].max())
    cmap_g = LinearSegmentedColormap.from_list(
        "g", ["#D1FAE5", "#34D399", ACCENT2, "#065F46"])
    colors = [cmap_g(norm(v)) for v in df["productivity_index"]]

    bars = ax.barh(df["Location"], df["productivity_index"],
                   color=colors, height=0.65, zorder=3,
                   edgecolor=BG, linewidth=0.4)
    ax.set_xlabel("Productivity index", color=MUTED)
    _spine_clean(ax)

    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{row.productivity_index:.2f}  |  {row.expert_pct:.0f}% experts",
                va="center", fontsize=8, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q12_geo_productivity.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q13  Trajectory Validation — one chart per trajectory type
# ─────────────────────────────────────────────────────────────────────────────
def chart_trajectory_validation():
    df = pd.read_csv("results/Q13_trajectory_validation.csv")
    traj_color = {"rising": ACCENT2, "stable": ACCENT1, "declining": ACCENT3}

    for traj in df["Trajectory"].unique():
        sub_traj = df[df["Trajectory"] == traj]
        col      = traj_color.get(traj, MUTED)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"Tag Trajectory  —  {traj.capitalize()} Tags",
                     fontsize=15, fontweight="bold", color=TEXT,
                     x=0.05, ha="left")
        _subtitle(ax, "% change in question volume relative to each tag's first recorded year")

        for tag in sub_traj["tag"].unique():
            sub = sub_traj[sub_traj["tag"] == tag].sort_values("yr")
            ax.plot(sub["yr"], sub["pct_change_from_base"],
                    color=col, linewidth=1.6, marker="o", markersize=4,
                    alpha=0.75, zorder=3,
                    path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
            last = sub.iloc[-1]
            ax.text(last["yr"] + 0.1, last["pct_change_from_base"],
                    tag, va="center", fontsize=8, color=col)

        ax.axhline(0, color=BORDER, linewidth=1, linestyle="--")
        ax.set_xlabel("Year", color=MUTED)
        ax.set_ylabel("% change from base year", color=MUTED)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        _spine_clean(ax)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(f"Q13_trajectory_{traj}.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q14  Power User Leaderboard — bubble scatter
# ─────────────────────────────────────────────────────────────────────────────
def chart_power_user_leaderboard():
    df = pd.read_csv("results/Q14_power_user_leaderboard.csv").head(30)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Power User Leaderboard", fontsize=15, fontweight="bold",
                 color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "Bubble size = badge points  |  Colour = platform value score  |  Axes = activity")

    norm   = plt.Normalize(df["platform_value_score"].min(),
                           df["platform_value_score"].max())
    cmap_p = LinearSegmentedColormap.from_list(
        "p", ["#EDE9FE", ACCENT4, "#3B0764"])
    colors = [cmap_p(norm(v)) for v in df["platform_value_score"]]
    sizes  = (df["badge_points"] / df["badge_points"].max() * 700).clip(lower=30)

    ax.scatter(df["questions"], df["accepted_answers"],
               s=sizes, c=colors, alpha=0.85, zorder=4,
               edgecolors=BORDER, linewidths=0.6)

    for _, row in df.nlargest(10, "platform_value_score").iterrows():
        ax.annotate(
            row["DisplayName"],
            xy=(row["questions"], row["accepted_answers"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, color=TEXT,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)],
        )

    ax.set_xlabel("Questions posted", color=MUTED)
    ax.set_ylabel("Accepted answers", color=MUTED)
    _spine_clean(ax)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap_p),
        ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Platform value score", color=MUTED, fontsize=8)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q14_power_user_leaderboard.png", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Q15  Self-Answer Rate — lollipop with mean line
# ─────────────────────────────────────────────────────────────────────────────
def chart_self_answer_rate():
    df = pd.read_csv("results/Q15_self_answer_rate.csv")
    df = df.sort_values("self_answer_rate_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("Self-Answer Rate by Tag", fontsize=15, fontweight="bold",
                 color=TEXT, x=0.05, ha="left")
    _subtitle(ax, "% of answered questions where the asker also wrote the accepted answer")

    x     = np.arange(len(df))
    rates = df["self_answer_rate_pct"].values
    mean  = rates.mean()
    cols  = [ACCENT3 if r > mean else ACCENT1 for r in rates]

    ax.vlines(x, 0, rates, colors=BORDER, linewidth=2)
    ax.scatter(x, rates, color=cols, s=90, zorder=4)
    ax.axhline(mean, color=MUTED, linestyle="--", linewidth=1.2,
               label=f"Mean  {mean:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(df["tag"], rotation=35, ha="right",
                       fontsize=9, color=TEXT)
    ax.set_ylabel("Self-answer rate (%)", color=MUTED)
    ax.set_ylim(0, rates.max() * 1.18)
    ax.legend(fontsize=9)
    _spine_clean(ax)

    ax.annotate(
        f"{rates[0]:.1f}%",
        xy=(0, rates[0]),
        xytext=(1.2, rates[0] + 0.8),
        fontsize=9, color=ACCENT3, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save("Q15_self_answer_rate.png", fig)



CHARTS = [
    ("Q01  Tag momentum",              chart_tag_momentum),
    ("Q02  Response-time percentiles", chart_response_percentiles),
    ("Q03  Monthly rolling avg",       chart_monthly_rolling),
    ("Q04  Engagement funnel",         chart_engagement_funnel),
    ("Q06  YoY bump chart",            chart_yoy_bump),
    ("Q08  Activity heatmap",          chart_activity_heatmap),
    ("Q09  Cohort retention",          chart_cohort_retention),
    ("Q10  Answer quality x tier",     chart_answer_quality_tiers),
    ("Q11  Badge velocity",            chart_badge_velocity),
    ("Q12  Geo productivity",          chart_geo_productivity),
    ("Q13  Trajectory validation",     chart_trajectory_validation),
    ("Q14  Power user leaderboard",    chart_power_user_leaderboard),
    ("Q15  Self-answer rate",          chart_self_answer_rate),
]


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[visualize] Generating charts ...\n")
    failed = []
    for label, fn in CHARTS:
        try:
            fn()
        except Exception as e:
            print(f"  x  {label}: {e}")
            failed.append(label)
    print(f"\n[visualize] Done.  "
          f"{len(CHARTS) - len(failed)}/{len(CHARTS)} succeeded.\n")
    if failed:
        print("  Failed:", failed)


if __name__ == "__main__":
    run()