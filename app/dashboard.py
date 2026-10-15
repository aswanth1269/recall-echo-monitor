"""Recall Echo Monitor dashboard.

Run:  .venv/Scripts/python.exe -m streamlit run app/dashboard.py

Reads only files the pipeline writes, through app/data_sources.py, which
lists every file and column it relies on. Nothing is hardcoded; a missing file
shows an empty state that names the command to run, and a file in an
unexpected shape says which columns are missing. Rates the pipeline left
empty (n < 10) are never shown, not even on hover.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import data_sources as ds  # noqa: E402
import pipeline_3d  # noqa: E402

MIN_N = 10  # rates with fewer campaigns than this are shown as "too few to compare"

# Dark chart palette (validated dark-mode steps of the reference palette).
INK, INK_2, MUTED = "#f0efec", "#c3c2b7", "#898781"
SURFACE, GRID, AXIS = "#1a1a19", "#2c2c2a", "#383835"
US, INDIA, MATCH = "#3987e5", "#d95926", "#199e70"
SMALL_N = "#4a4a47"
DASH = "\u2014"

st.set_page_config(page_title="Recall Echo Monitor", page_icon=":material/sync_alt:",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .hero {{ display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
           padding: 4px 0 18px; border-bottom: 1px solid rgba(255,255,255,.08); margin-bottom: 18px; }}
  .hero h1 {{ font-size: 2.1rem; margin: 0; font-weight: 650; }}
  .hero p {{ margin: 6px 0 0; color: {INK_2}; font-size: 1.02rem; max-width: 720px; }}
  .hero .asof {{ color: {MUTED}; font-size: .85rem; text-align: right; white-space: nowrap; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
           border:1px solid rgba(255,255,255,.12); color:{INK_2}; margin-right:6px; }}
  .tiles {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; margin: 6px 0 22px; }}
  @media (max-width: 900px) {{ .tiles {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
  .tile {{ background:{SURFACE}; border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:16px 18px; }}
  .tile .k {{ color:{MUTED}; font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; }}
  .tile .v {{ font-size:2rem; font-weight:650; margin-top:6px; color:{INK}; }}
  .tile .s {{ color:{INK_2}; font-size:.85rem; margin-top:2px; }}
  .tile .bar {{ height:3px; border-radius:3px; margin-top:12px; background:{GRID}; overflow:hidden; }}
  .tile .bar div {{ height:100%; border-radius:3px; }}
  .card {{ background:{SURFACE}; border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:16px 18px; }}
  .note {{ color:{MUTED}; font-size:.85rem; }}
  mark {{ background: rgba(25,158,112,.28); color:{INK}; padding:0 2px; border-radius:3px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
  .stTabs [data-baseweb="tab"] {{ padding: 8px 14px; border-radius: 10px 10px 0 0; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading ---
@st.cache_data(show_spinner=False)
def _load(name: str, stamp: tuple) -> pd.DataFrame | None:
    # `stamp` is every output file's mtime: any pipeline rewrite gives a new
    # cache key, so "run the command and refresh" really shows the new file.
    return getattr(ds, name)()


def data(name: str) -> pd.DataFrame | None:
    """A dashboard table from data_sources, or None (with a note if malformed)."""
    try:
        return _load(name, ds.stamp())
    except ds.FormatError as exc:
        st.markdown(f"<div class='card note'>{html.escape(str(exc))}</div>",
                    unsafe_allow_html=True)
        return None


def empty_state(what: str, command: str) -> None:
    st.markdown(f"<div class='card note'>No {what} yet. Run <code>{command}</code> "
                "and refresh this page.</div>", unsafe_allow_html=True)


def rows(df: pd.DataFrame | None, group_type: str) -> pd.DataFrame:
    if df is None or "group_type" not in df:
        return pd.DataFrame()
    return df[df["group_type"].astype(str).str.lower() == group_type].copy()


def retrieved_dates() -> str:
    src = ds.root() / "docs/SOURCES.md"
    if not src.exists():
        return ""
    dates = re.findall(r"\| Retrieved \| (\d{4}-\d{2}-\d{2}) \|", src.read_text(encoding="utf-8"))
    return f"Data retrieved {max(dates)}" if dates else ""


def pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{float(v) * 100:.1f}%"


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        hoverlabel=dict(bgcolor="#232321", bordercolor="#383835", font=dict(color=INK)),
        showlegend=False, bargap=0.55)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=INK_2))
    return fig


def rate_bars(df: pd.DataFrame, label_col: str, height: int = 320) -> go.Figure:
    """Horizontal echo-rate bars, sorted, n on every bar.

    Groups with n < MIN_N (or no rate from the pipeline) get no bar and no rate
    anywhere - label and hover say "too few to compare" - per CLAUDE.md.
    """
    d = df.copy()
    d["small"] = (d["n"] < MIN_N) | d["echo_rate"].isna()
    d = d.sort_values(["small", "echo_rate"], ascending=[False, True])  # small n at the bottom, best at the top
    colors = [SMALL_N if s else US for s in d["small"]]
    text = [f"n={n}  (too few to compare)" if s else f"{r * 100:.0f}%  ·  n={n}"
            for r, n, s in zip(d["echo_rate"], d["n"], d["small"])]
    hover = [f"<b>{html.escape(str(y))}</b><br>n={n}: too few campaigns to show a rate" if s else
             f"<b>{html.escape(str(y))}</b><br>Echo rate {r:.1%}<br>{int(e)} of {n} campaigns"
             for y, r, n, e, s in zip(d[label_col], d["echo_rate"], d["n"], d["echoing"], d["small"])]
    fig = go.Figure(go.Bar(
        x=[0 if s else r for r, s in zip(d["echo_rate"], d["small"])],
        y=d[label_col].astype(str), orientation="h",
        marker=dict(color=colors, cornerradius=4), text=text, textposition="outside",
        textfont=dict(color=INK_2, size=12), cliponaxis=False,
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>"))
    shown = d.loc[~d["small"], "echo_rate"]
    fig.update_xaxes(tickformat=".0%",
                     range=[0, max(1.0, float(shown.max()) * 1.25 if len(shown) else 1.0)])
    return style(fig, height)


# ------------------------------------------------------------------- hero ---
st.markdown(f"""
<div class="hero">
  <div>
    <span class="pill">NHTSA (US)</span><span class="pill">SIAM (India)</span><span class="pill">LLM + evidence check</span>
    <h1>Recall Echo Monitor</h1>
    <p>When a carmaker recalls a model in the US for a defect, does a matching recall reach India, and how late?</p>
  </div>
  <div class="asof">{html.escape(retrieved_dates())}</div>
</div>""", unsafe_allow_html=True)

tab_overview, tab_how, tab_explore, tab_method = st.tabs(
    ["Overview", "How it works", "Recall explorer", "Method & limits"])

# ---------------------------------------------------------------- overview ---
with tab_overview:
    rate, lag = data("echo_rate"), data("echo_lag")
    ota, evals = data("ota_share"), data("eval_summary")

    o_rate, o_lag, o_ota = rows(rate, "overall"), rows(lag, "overall"), rows(ota, "overall")
    llm = evals[evals["matcher"].astype(str).str.lower() == "llm"] if evals is not None else pd.DataFrame()

    def tile(k: str, v: str, s: str, share: float | None, color: str) -> str:
        width = 0 if share is None or pd.isna(share) else max(0, min(100, share * 100))
        return (f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div>"
                f"<div class='s'>{s}</div><div class='bar'><div style='width:{width:.1f}%;background:{color}'></div></div></div>")

    r = o_rate.iloc[0] if not o_rate.empty else None
    l = o_lag.iloc[0] if not o_lag.empty else None
    o = o_ota.iloc[0] if not o_ota.empty else None
    e = llm.iloc[0] if not llm.empty else None
    st.markdown("<div class='tiles'>" + "".join([
        tile("Echo rate", pct(r["echo_rate"]) if r is not None else "—",
             f"{int(r['echoing'])} of {int(r['n'])} US campaigns on India-sold models" if r is not None else "run src.kpis",
             r["echo_rate"] if r is not None else None, MATCH),
        tile("Median echo lag", f"{l['median_days']:.0f} days" if l is not None and pd.notna(l["median_days"]) else "—",
             f"middle half {l['p25_days']:.0f} to {l['p75_days']:.0f} days · n={int(l['n'])}" if l is not None and pd.notna(l["p25_days"]) else "run src.kpis",
             None, MATCH),
        tile("Over-the-air share", pct(o["ota_share"]) if o is not None else "—",
             f"{int(o['ota_n'])} of {int(o['n'])} campaigns with a known flag (API flag; a lower bound)"
             if o is not None else "run src.fetch_ota then src.kpis",
             o["ota_share"] if o is not None else None, US),
        tile("Matcher F1",
             (f"{e['f1']:.2f}" if pd.notna(e["f1"]) else "n<10") if e is not None else "—",
             (f"found {int(e['tp'])} of {int(e['positives'])} echoes · {int(e['fp'])} false · "
              f"{int(e['n'])} test pairs, labelled by Claude and tuned on")
             if e is not None else "run src.evaluate",
             e["f1"] if e is not None and pd.notna(e["f1"]) else None, INK_2),
    ]) + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.subheader("Echo rate by carmaker")
        mk = rows(rate, "make")
        if mk.empty:
            empty_state("echo rates", "python -m src.kpis")
        else:
            st.plotly_chart(rate_bars(mk, "group", height=40 * len(mk) + 60), width="stretch")
            st.caption(f"Grey bars have fewer than {MIN_N} campaigns, too few to compare fairly.")
    with c2:
        st.subheader("Luxury vs mass-market")
        seg = rows(rate, "segment")
        if seg.empty:
            empty_state("segment comparison", "python -m src.kpis")
        else:
            seg["group"] = seg["group"].astype(str).str.replace("_", "-").str.capitalize()
            st.plotly_chart(rate_bars(seg, "group", height=200), width="stretch")
        seg_lag = rows(lag, "segment")
        if not seg_lag.empty:
            for _, s in seg_lag.iterrows():
                st.markdown(f"<span class='note'>{html.escape(str(s['group']).replace('_', '-'))}: median lag "
                            f"{s['median_days']:.0f} days (n={int(s['n'])}), India first in {int(s['india_first'])}</span>",
                            unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.subheader("How long until India recalls")
        ml = rows(lag, "make")
        if ml.empty:
            empty_state("lag figures", "python -m src.kpis")
        else:
            ml = ml.dropna(subset=["median_days"]).sort_values("median_days")
            fig = go.Figure(go.Scatter(
                x=ml["median_days"], y=ml["group"].astype(str), mode="markers",
                marker=dict(size=11, color=MATCH, line=dict(color=SURFACE, width=2)),
                error_x=dict(type="data", symmetric=False,
                             array=ml["p75_days"] - ml["median_days"],
                             arrayminus=ml["median_days"] - ml["p25_days"],
                             color=INK_2, thickness=2, width=0),
                customdata=ml[["p25_days", "p75_days", "n"]],
                hovertemplate="<b>%{y}</b><br>Median %{x:.0f} days<br>Middle half %{customdata[0]:.0f} to %{customdata[1]:.0f} days<br>n=%{customdata[2]}<extra></extra>"))
            fig.add_vline(x=0, line_color=AXIS, line_width=1)
            fig.update_xaxes(title_text="Days from US recall to India recall (negative = India first)",
                             title_font=dict(color=MUTED, size=12))
            st.plotly_chart(style(fig, 40 * len(ml) + 90), width="stretch")
            st.caption("Dot = median. Line = middle half of matched recalls.")
    with c4:
        st.subheader("Which defects echo least")
        comp = data("components")
        if comp is None or comp.empty:
            empty_state("component figures", "python -m src.kpis")
        else:
            comp = comp[comp["n"] >= MIN_N].nsmallest(10, "echo_rate")
            st.plotly_chart(rate_bars(comp, "component_group", height=36 * len(comp) + 60), width="stretch")
            st.caption(f"The 10 component groups with the lowest echo rate, among groups with n ≥ {MIN_N}.")

    oe = data("ota_echo")
    if oe is not None and not oe.empty:
        st.subheader("Do over-the-air fixes echo differently?")
        st.plotly_chart(rate_bars(oe, "ota", height=190), width="stretch")
        st.caption("OTA = the NHTSA API flags the remedy as over-the-air. The flag misses some "
                   "OTA remedies described only in the remedy text, so OTA counts are a lower bound.")

# ------------------------------------------------------------ how it works ---
with tab_how:
    pipeline_3d.render(height=620)
    st.markdown("<p class='note'>Counts in the animation are read live from the pipeline's database. "
                "A dash means that stage has not been run yet.</p>", unsafe_allow_html=True)

# --------------------------------------------------------------- explorer ---
with tab_explore:
    ex = data("explorer")
    if ex is None or ex.empty:
        empty_state("explorer data", "python -m src.kpis")
    else:
        f1, f2, f3 = st.columns([2, 2, 3])
        makes = f1.multiselect("Carmaker", sorted(ex["make"].dropna().unique()))
        labels = f2.multiselect("Result", ["same", "different", "needs_review", "no_candidate"],
                                format_func={"same": "India echo found", "different": "No match",
                                             "needs_review": "Needs review",
                                             "no_candidate": "No India candidate"}.get)
        query = f3.text_input("Search component or defect", placeholder="e.g. airbag, software, brake")
        view = ex.copy()
        if makes:
            view = view[view["make"].isin(makes)]
        if labels:
            view = view[view["label"].isin(labels)]
        if query:
            q = query.lower()
            view = view[view["component"].astype(str).str.lower().str.contains(q, regex=False)
                        | view["us_defect"].astype(str).str.lower().str.contains(q, regex=False)]

        st.markdown(f"<span class='note'>{len(view):,} of {len(ex):,} US campaigns. "
                    "“No India candidate” is a transparency signal, not a finding of wrongdoing.</span>",
                    unsafe_allow_html=True)
        nice = {"same": "India echo found", "different": "No match", "needs_review": "Needs review",
                "no_candidate": "No India candidate"}
        table = view.assign(result=view["label"].map(nice).fillna(view["label"]))
        cols = ["CAMPNO", "make", "report_date", "component", "result", "india_recall_date", "lag_days", "confidence"]
        event = st.dataframe(
            table[cols], hide_index=True, width="stretch", height=420,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "CAMPNO": "US campaign", "make": "Carmaker", "report_date": "US date",
                "component": st.column_config.TextColumn("Component", width="large"),
                "result": "Result", "india_recall_date": "India date",
                "lag_days": st.column_config.NumberColumn("Lag (days)", format="%d"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            })

        sel = event.selection.rows if event and event.selection else []
        if sel:
            row = view.iloc[sel[0]]

            def highlight(text: object, phrase: object) -> str:
                """Escape text and <mark> the evidence, found the way the matcher checks it."""
                text = ds.text(text)
                span = ds.find_evidence(text, phrase)
                if span is None:
                    return html.escape(text)
                i, j = span
                return (html.escape(text[:i]) + "<mark>" + html.escape(text[i:j]) + "</mark>"
                        + html.escape(text[j:]))

            label = ds.text(row["label"])
            a, b = st.columns(2, gap="large")
            a.markdown(f"<div class='card'><div class='note'>US · {html.escape(ds.text(row['CAMPNO']))} · "
                       f"{html.escape(ds.text(row['report_date']))}</div><h4 style='margin:.3rem 0'>{html.escape(ds.text(row['component']) or DASH)}</h4>"
                       f"<p>{highlight(row['us_defect'], row.get('us_evidence'))}</p></div>", unsafe_allow_html=True)
            b.markdown(f"<div class='card'><div class='note'>India · {html.escape(ds.text(row.get('india_model')) or DASH)} · "
                       f"{html.escape(ds.text(row.get('india_recall_date')) or DASH)}</div><h4 style='margin:.3rem 0'>{html.escape(nice.get(label, label))}</h4>"
                       f"<p>{highlight(row.get('india_defect'), row.get('india_evidence')) or DASH}</p></div>", unsafe_allow_html=True)
            if label in ("same", "different"):
                st.caption("Highlighted text is the evidence the LLM quoted; Python confirmed both "
                           "quotes exist in the source text. The verdict itself is the LLM's and has "
                           "not been reviewed by a person.")
            elif label == "needs_review":
                st.caption("Needs review: the LLM's answer could not be confirmed (a quote was missing "
                           "from its source, or there was no defect text to judge).")

# ----------------------------------------------------------------- method ---
with tab_method:
    m1, m2 = st.columns([3, 2], gap="large")
    with m1:
        st.subheader("Pipeline")
        st.markdown("""
1. **Load** NHTSA US recalls and SIAM India recalls into DuckDB. Unreadable dates are logged, never guessed.
2. **Map models** on both sides to one family list: rules, then a batched LLM, then an optional overrides file a person maintains. LLM-mapped names are listed for review.
3. **Find candidates**: same carmaker, shared family, India recall from 180 days before to 730 days after the US one, top 3 by TF-IDF similarity.
4. **Match**: an LLM labels each pair and quotes evidence from both texts; Python checks each quote exists, or the pair goes to review.
5. **Measure**: echo rate, lag, gaps by component and over-the-air share, each with its n.
""")
        st.subheader("Limits")
        st.markdown("""
- India model lists are today's line-ups, and some rows come from secondary sources.
- SIAM lists voluntary recalls only, and Indian variants can differ from US ones.
- Some carmakers have very few India recalls, so their rates are not compared.
- Model-family mappings made by the LLM, and the matcher's same/different verdicts, have not yet been reviewed by a person.
- The over-the-air flag comes from the NHTSA API. It misses some remedies whose text says over-the-air, so OTA counts are a lower bound.
- A missing India recall is a transparency signal, not proof of wrongdoing.
""")
    with m2:
        st.subheader("Matcher accuracy")
        if evals is None or evals.empty:
            empty_state("evaluation results", "python -m src.evaluate")
        else:
            show = evals[["matcher", "n", "positives", "tp", "fp", "fn", "tn",
                          "precision", "recall", "f1"]].copy()
            show["matcher"] = show["matcher"].map({"llm": "LLM + evidence check", "baseline": "TF-IDF baseline"}).fillna(show["matcher"])
            st.dataframe(show, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(c.capitalize(), format="%.2f")
                                        for c in ("precision", "recall", "f1")})
            st.caption(f"Agreement with {int(evals['n'].iloc[0])} test pairs labelled by Claude, "
                       "not yet checked by a person. The LLM prompt and the baseline threshold were "
                       f"both tuned on these same pairs, so this is not an accuracy estimate. Rates "
                       f"with fewer than {MIN_N} cases are left blank; the counts are all there.")
        st.subheader("Sources")
        st.markdown(f"""
- [NHTSA recall data](https://www.nhtsa.gov/nhtsa-datasets-and-apis), US public domain
- [SIAM voluntary recalls](https://www.siam.in/siam-voluntary-recall.aspx), saved snapshot
- {html.escape(retrieved_dates()) or 'See docs/SOURCES.md'}
""")
