"""
SENTINEL — ESA Spacecraft Telemetry Anomaly Detection Dashboard
================================================================
    pip install streamlit requests plotly pandas
    streamlit run sentinel_app.py
"""
import time
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

API_BASE          = "https://sentinel-841294993537.europe-west1.run.app"
MISSION_START     = datetime(2007, 1, 1, 0, 0, 30)
SAMPLE_INTERVAL_S = 30

GREEN     = "#33ff33"
GREEN_DIM = "#1a8c1a"
AMBER     = "#ffb000"
RED       = "#ff3333"
RED_BG    = "rgba(255,51,51,0.08)"
CYAN      = "#00cccc"
BG        = "#0a0a0a"
PANEL     = "#0f0f0f"
GRID      = "#1e1e1e"
MONO      = "'IBM Plex Mono', 'Courier New', monospace"

CH_COLOURS = [GREEN, AMBER, CYAN, "#ff6633", "#cc33ff",
              "#33ccff", "#ffcc00", "#ff3399", "#66ff66", "#ff9933"]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def id_to_dt(obs_id: int) -> datetime:
    return MISSION_START + timedelta(seconds=int(obs_id) * SAMPLE_INTERVAL_S)

def dt_to_id(dt: datetime) -> int:
    return max(0, int((dt - MISSION_START).total_seconds() / SAMPLE_INTERVAL_S))

def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(MISSION_START) + pd.to_timedelta(df["id"] * SAMPLE_INTERVAL_S, unit="s")
    return df

def find_bands(df: pd.DataFrame) -> list[tuple]:
    a = df["is_anomaly"].values
    if a.sum() == 0: return []
    padded = np.concatenate(([0], a, [0]))
    d = np.diff(padded)
    ts = df["timestamp"].values
    return [(ts[s], ts[e]) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1)]

def cluster_lengths(series: pd.Series) -> list[int]:
    a = series.values
    if a.sum() == 0: return []
    padded = np.concatenate(([0], a, [0]))
    d = np.diff(padded)
    return (np.where(d == -1)[0] - np.where(d == 1)[0]).tolist()

def add_anomaly_bands(fig: go.Figure, bands: list[tuple]):
    for s, e in bands:
        fig.add_vrect(x0=s, x1=e, fillcolor=RED_BG, line_width=0)

def downsample_max(values, timestamps, block=500):
    n = len(values)
    if n <= block * 2: return timestamps, values
    nf = (n // block) * block
    ds_v = values[:nf].reshape(-1, block).max(axis=1)
    ds_t = timestamps[:nf:block]
    if nf < n:
        ds_v = np.append(ds_v, values[nf:].max())
        ds_t = np.append(ds_t, timestamps[nf])
    return ds_t, ds_v

def retro_layout(height=300, margin=None, **kw) -> dict:
    ax = dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    return dict(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        font=dict(family=MONO, size=13, color=GREEN),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=GREEN_DIM)),
        dragmode="zoom", height=height, margin=margin or dict(l=40, r=20, t=10, b=40),
        xaxis={**ax, **kw.get("xaxis", {})}, yaxis={**ax, **kw.get("yaxis", {})},
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="SENTINEL", page_icon="S", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {{ background-color: {BG} !important; font-family: {MONO} !important; }}
[data-testid="stAppViewContainer"]::after {{ content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0; pointer-events: none; z-index: 9999; background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px); }}
.block-container {{ padding-top: 1.5rem; max-width: 1400px; }}
h1,h2,h3,h4,h5,h6,p,label {{ font-family: {MONO} !important; color: {GREEN} !important; }}
h2 {{ letter-spacing: 0.3em !important; text-transform: uppercase !important; text-shadow: 0 0 10px {GREEN}; }}
h4 {{ letter-spacing: 0.2em !important; text-transform: uppercase !important; color: {AMBER} !important; border-bottom: 1px solid {GRID}; padding-bottom: 8px; }}
h5 {{ letter-spacing: 0.15em !important; text-transform: uppercase !important; color: {AMBER} !important; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 0; padding: 16px 20px; }}
div[data-testid="stMetric"] label {{ font-family: {MONO} !important; font-size: 12px !important; letter-spacing: 0.12em; text-transform: uppercase; color: {GREEN_DIM} !important; }}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ font-family: {MONO} !important; color: {GREEN} !important; text-shadow: 0 0 8px rgba(51,255,51,0.4); }}
[data-testid="stTabs"] button {{ font-family: {MONO} !important; letter-spacing: 0.12em; text-transform: uppercase; color: {GREEN_DIM} !important; border-radius: 0 !important; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: {GREEN} !important; border-bottom: 2px solid {GREEN} !important; }}
[data-testid="stButton"] button {{ font-family: {MONO} !important; letter-spacing: 0.1em; text-transform: uppercase; border-radius: 0 !important; border: 1px solid {GREEN} !important; background: rgba(51,255,51,0.06) !important; color: {GREEN} !important; }}
.status-ok {{ display:inline-block; font-family:{MONO}; font-size:14px; padding:6px 16px; border:1px solid {GREEN}; color:{GREEN}; background:rgba(51,255,51,0.04); letter-spacing:0.15em; }}
.status-fail {{ display:inline-block; font-family:{MONO}; font-size:14px; padding:6px 16px; border:1px solid {RED}; color:{RED}; background:{RED_BG}; letter-spacing:0.15em; }}
.boot {{ font-family: {MONO}; font-size: 15px; color: {GREEN_DIM}; line-height: 1.8; }}
.boot .w {{ color: {AMBER}; }}
.boot .r {{ color: {RED}; }}
.boot .g {{ color: {GREEN}; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_health() -> dict:
    try: return requests.get(f"{API_BASE}/", timeout=5).json()
    except Exception: return {}

@st.cache_data(ttl=300)
def fetch_timeline() -> pd.DataFrame:
    return pd.DataFrame(requests.get(f"{API_BASE}/timeline", timeout=60).json())

@st.cache_data(ttl=600)
def fetch_features() -> list[str]:
    try:
        r = requests.get(f"{API_BASE}/features", timeout=5)
        if r.ok:
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], str): return data
    except Exception: pass
    return []

@st.cache_data(ttl=300)
def fetch_channel(channel: str, start: int, end: int) -> pd.DataFrame:
    r = requests.get(f"{API_BASE}/channels",
                     params={"channel": channel, "start": start, "end": end}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

@st.cache_data(ttl=600)
def fetch_report() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/report", timeout=60)
        return r.json() if r.ok else None
    except Exception: return None


# ═══════════════════════════════════════════════════════════════════════════════
#  LOADING SCREEN — text only, does the actual data fetching
# ═══════════════════════════════════════════════════════════════════════════════

if "loaded" not in st.session_state:
    st.markdown("## SENTINEL")
    log = st.empty()
    lines: list[str] = []

    def msg(text, cls=""):
        tag = f' class="{cls}"' if cls else ""
        lines.append(f"<span{tag}>&gt; {text}</span>")
        log.markdown(f'<div class="boot">{"<br>".join(lines)}</div>', unsafe_allow_html=True)
        time.sleep(0.15)

    msg("SENTINEL V1.0.0 // PCA RESIDUAL MODEL")
    msg(f"CONNECTING TO {API_BASE}")

    health = fetch_health()
    if health.get("status") == "ok":
        msg("UPLINK ESTABLISHED", "g")
    else:
        msg("UPLINK FAILED", "r")
        st.stop()

    msg("LOADING TIMELINE DATA...")
    try:
        df = add_timestamps(fetch_timeline())
        n_anom = df["is_anomaly"].tolist().count(1)
        msg(f"RECEIVED {len(df):,} OBSERVATIONS // {n_anom:,} ANOMALIES DETECTED", "g")
    except Exception as e:
        msg(f"TIMELINE FAILED: {e}", "r")
        st.stop()

    msg("LOADING CHANNEL LIST...")
    features = fetch_features()
    if features:
        msg(f"{len(features)} CHANNELS AVAILABLE", "g")
    else:
        msg("NO CHANNELS FOUND", "w")

    msg("SYSTEM READY", "w")
    time.sleep(0.5)

    st.session_state.loaded = True
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA (runs after loading screen, everything is cached)
# ═══════════════════════════════════════════════════════════════════════════════

df = add_timestamps(fetch_timeline())

# Pure Python — numpy int/format is broken on Python 3.14
_labels    = df["is_anomaly"].tolist()
total      = len(_labels)
anomalies  = _labels.count(1)
normal     = total - anomalies
rate       = round(anomalies / total * 100, 2) if total else 0.0
n_clusters = sum(1 for i in range(1, total) if _labels[i] == 1 and _labels[i-1] == 0)

# DEBUG — remove once deployed
import sys
print(f"DEBUG types: total={type(total).__name__}({total}), anomalies={type(anomalies).__name__}({anomalies}), normal={type(normal).__name__}({normal})", file=sys.stderr)
c_lengths  = cluster_lengths(df["is_anomaly"])
bands      = find_bands(df)
min_dt     = MISSION_START
max_dt     = id_to_dt(total - 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

col_t, col_s = st.columns([4, 1])
col_t.markdown("## SENTINEL")
col_t.caption("ESA SPACECRAFT TELEMETRY // ANOMALY DETECTION")
api_ok = fetch_health().get("status") == "ok"
col_s.markdown(f'<div class="status-{"ok" if api_ok else "fail"}">{"SYS NOMINAL" if api_ok else "LINK DOWN"}</div>', unsafe_allow_html=True)

tab_overview, tab_channels, tab_model, tab_log = st.tabs(
    ["OVERVIEW", "CHANNEL EXPLORER", "MODEL REPORT", "OBSERVATION LOG"]
)

# ── OVERVIEW ─────────────────────────────────────────────────────────────────

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL OBSERVATIONS", f"{int(total):,}")
    c2.metric("ANOMALIES DETECTED", f"{int(anomalies):,}", delta=f"{rate:.2f}%", delta_color="inverse")
    c3.metric("NOMINAL", f"{int(normal):,}")
    c4.metric("CLUSTERS", f"{int(n_clusters):,}")

    if anomalies > 0:
        st.divider()
        st.markdown("#### ANOMALY DENSITY")
        anom_df = df[df["is_anomaly"] == 1]
        daily = anom_df.groupby(anom_df["timestamp"].dt.date).size().reset_index(name="count")
        bar = go.Figure(go.Bar(x=daily["timestamp"], y=daily["count"], marker_color=RED, opacity=0.7))
        bar.update_layout(**retro_layout(200, margin=dict(l=40, r=20, t=10, b=30),
            xaxis=dict(title="DATE"), yaxis=dict(title="ANOMALIES / DAY")))
        st.plotly_chart(bar, width="stretch")

        st.divider()
        st.markdown("#### ANOMALY CLUSTERS")
        cluster_rows = []
        for i, ((ts_s, ts_e), length) in enumerate(zip(bands, c_lengths)):
            s, e = pd.Timestamp(ts_s), pd.Timestamp(ts_e)
            sec = int((e - s).total_seconds())
            dur = f"{sec//3600}h {(sec%3600)//60}m" if sec >= 3600 else f"{sec//60}m {sec%60}s" if sec >= 60 else f"{sec}s"
            cluster_rows.append({"CLUSTER": i+1, "START": s.strftime("%Y-%m-%d %H:%M"),
                "END": e.strftime("%Y-%m-%d %H:%M"), "DURATION": dur, "POINTS": length})
        st.dataframe(pd.DataFrame(cluster_rows), width="stretch", hide_index=True)

# ── CHANNEL EXPLORER ─────────────────────────────────────────────────────────

with tab_channels:
    st.markdown("#### CHANNEL EXPLORER")
    st.caption("SIGNAL TURNS RED DURING ANOMALY DETECTIONS")

    known = fetch_features()
    col_ch, col_from, col_to = st.columns([3, 1.5, 1.5])
    with col_ch:
        if known:
            selected = st.multiselect("CHANNELS", known, default=known[:1])
        else:
            raw = st.text_input("CHANNELS", placeholder="channel_41, channel_42")
            selected = [c.strip() for c in raw.split(",") if c.strip()] if raw else []

    with col_from:
        from_date = st.date_input("FROM", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
    with col_to:
        to_date = st.date_input("TO", value=min(min_dt + timedelta(days=7), max_dt).date(),
                                min_value=min_dt.date(), max_value=max_dt.date())

    id_start = dt_to_id(datetime.combine(from_date, datetime.min.time()))
    id_end   = min(dt_to_id(datetime.combine(to_date, datetime.max.time())), total - 1)

    st.caption(f"ID {int(id_start):,} — {int(id_end):,} // {int(id_end - id_start + 1):,} POINTS")
    go_btn = st.button("FETCH", type="primary")

    if go_btn and selected:
        ch_data: dict[str, pd.DataFrame] = {}
        prog = st.progress(0, text="ACQUIRING TELEMETRY...")
        for i, name in enumerate(selected):
            try: ch_data[name] = add_timestamps(fetch_channel(name, id_start, id_end))
            except Exception as e: st.warning(f"{name}: {e}")
            prog.progress((i + 1) / len(selected), text=f"ACQUIRED {name}")
        prog.empty()

        if ch_data:
            cols = st.columns(min(len(ch_data), 4))
            for i, (name, cdf) in enumerate(ch_data.items()):
                cols[i % len(cols)].metric(name.upper(), f"{cdf['value'].mean():.3f}",
                    delta=f"{int(cdf['is_anomaly'].sum())} ANOMALIES DETECTED")

            ch_fig = go.Figure()
            for i, (name, cdf) in enumerate(ch_data.items()):
                c = CH_COLOURS[i % len(CH_COLOURS)]
                is_anom = cdf["is_anomaly"].values.astype(bool)
                vals = cdf["value"].values.astype(float)

                normal = vals.copy(); normal[is_anom] = np.nan
                ch_fig.add_trace(go.Scattergl(x=cdf["timestamp"], y=normal,
                    mode="lines", line=dict(color=c, width=1.2),
                    name=name.upper(), connectgaps=False))

                if is_anom.any():
                    flagged = vals.copy(); flagged[~is_anom] = np.nan
                    ch_fig.add_trace(go.Scattergl(x=cdf["timestamp"], y=flagged,
                        mode="lines", line=dict(color=RED, width=2.5),
                        name=f"{name.upper()} ANOMALY", connectgaps=False))

            ch_fig.update_layout(**retro_layout(420, margin=dict(l=50, r=20, t=30, b=40),
                xaxis=dict(title="DATE"), yaxis=dict(title="SIGNAL")))
            st.plotly_chart(ch_fig, width="stretch")

            with st.expander("CHANNEL STATISTICS"):
                rows = [{"CHANNEL": n, "MEAN": round(d["value"].mean(), 4),
                         "STD": round(d["value"].std(), 4),
                         "MIN": round(d["value"].min(), 4), "MAX": round(d["value"].max(), 4),
                         "ANOMALIES": int(d["is_anomaly"].sum())} for n, d in ch_data.items()]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    elif go_btn:
        st.warning("SELECT AT LEAST ONE CHANNEL")

# ── MODEL REPORT ─────────────────────────────────────────────────────────────

with tab_model:
    st.markdown("#### MODEL REPORT")
    report = fetch_report()

    if report:
        features_list = report.get("features", [])
        mr1, mr2, mr3, mr4 = st.columns(4)
        if "threshold" in report:  mr1.metric("THRESHOLD", f"{report['threshold']:.6f}")
        if "n_anomalies" in report: mr2.metric("ANOMALIES", f"{int(report['n_anomalies']):,}")
        if features_list:           mr3.metric("FEATURES", f"{len(features_list)}")
        mr4.metric("WINDOW", "100 ROWS")

        st.divider()

        scores_raw = report.get("row_scores")
        if scores_raw and isinstance(scores_raw, list):
            st.markdown("##### RECONSTRUCTION ERROR")
            s = np.array(scores_raw, dtype=np.float32)
            ts_full = pd.to_datetime(MISSION_START) + pd.to_timedelta(np.arange(len(s)) * SAMPLE_INTERVAL_S, unit="s")
            thr = report.get("threshold", 0)

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("MEAN", f"{s.mean():.6f}")
            sc2.metric("MEDIAN", f"{np.median(s):.6f}")
            sc3.metric("P99", f"{np.percentile(s, 99):.6f}")
            sc4.metric("MAX", f"{s.max():.6f}")

            ts_ds, s_ds = downsample_max(s, ts_full, block=500)
            sfig = go.Figure()
            add_anomaly_bands(sfig, bands)
            sfig.add_trace(go.Scattergl(x=ts_ds, y=s_ds, mode="lines",
                line=dict(color=AMBER, width=1), name="ERROR"))
            sfig.add_hline(y=thr, line_dash="dash", line_color=RED, line_width=1,
                           annotation_text=f"THR={thr:.4f}", annotation_font_color=RED)
            sfig.update_layout(**retro_layout(300, margin=dict(l=50, r=20, t=10, b=40),
                xaxis=dict(title="DATE"), yaxis=dict(title="SCORE")))
            st.plotly_chart(sfig, width="stretch")

            st.markdown("##### SCORE DISTRIBUTION")
            hfig = go.Figure(go.Histogram(x=s, nbinsx=100, marker_color=AMBER, opacity=0.75))
            hfig.add_vline(x=thr, line_dash="dash", line_color=RED, line_width=1,
                           annotation_text="THRESHOLD", annotation_font_color=RED)
            hfig.update_layout(**retro_layout(250, margin=dict(l=40, r=20, t=10, b=30),
                xaxis=dict(title="RECONSTRUCTION ERROR"), yaxis=dict(title="FREQUENCY", type="log")))
            st.plotly_chart(hfig, width="stretch")

        per_ch = report.get("per_channel_mse")
        if per_ch and isinstance(per_ch, list):
            st.divider()
            st.markdown("##### PER-CHANNEL MSE")
            if isinstance(per_ch[0], dict):
                mse_df = pd.DataFrame(per_ch).rename(columns={"channel": "CHANNEL", "mse": "MSE"})
            else:
                mse_df = pd.DataFrame({"CHANNEL": features_list[:len(per_ch)], "MSE": per_ch})
            mse_df = mse_df.sort_values("MSE", ascending=False)
            mfig = go.Figure(go.Bar(x=mse_df["MSE"], y=mse_df["CHANNEL"],
                orientation="h", marker_color=AMBER, opacity=0.8))
            mfig.update_layout(**retro_layout(max(300, len(mse_df) * 22),
                margin=dict(l=130, r=20, t=10, b=30),
                xaxis=dict(title="MSE"), yaxis=dict(autorange="reversed")))
            st.plotly_chart(mfig, width="stretch")

        topk = report.get("window_top_channels")
        if topk and isinstance(topk, list) and features_list:
            st.divider()
            st.markdown("##### TOP CONTRIBUTING CHANNELS")
            flat = [features_list[i] if i < len(features_list) else f"idx_{i}"
                    for window in topk for i in window]
            freq_df = pd.Series(flat).value_counts().reset_index()
            freq_df.columns = ["CHANNEL", "APPEARANCES"]
            st.dataframe(freq_df.head(20), width="stretch", hide_index=True)

        if features_list:
            with st.expander("FEATURE LIST"):
                st.code(", ".join(features_list))

        with st.expander("RAW /REPORT"):
            st.json({k: (f"[{int(len(v)):,} values]" if isinstance(v, list) and len(v) > 20 else v)
                     for k, v in report.items()})
    else:
        st.warning("REPORT ENDPOINT UNAVAILABLE")

    st.divider()
    st.markdown("##### PREDICTION SUMMARY")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("TOTAL", f"{int(total):,}")
    mc2.metric("ANOMALOUS", f"{int(anomalies):,}")
    mc3.metric("NOMINAL", f"{int(normal):,}")
    mc4.metric("RATE", f"{rate:.2f}%")

    if n_clusters > 0:
        mc5, mc6, mc7, mc8 = st.columns(4)
        mc5.metric("CLUSTERS", f"{int(n_clusters):,}")
        mc6.metric("AVG LENGTH", f"{np.mean(c_lengths):.1f}")
        mc7.metric("MAX LENGTH", f"{int(max(c_lengths)):,}")
        mc8.metric("MIN LENGTH", f"{int(min(c_lengths)):,}")

    st.divider()
    st.markdown("##### ANOMALY DISTRIBUTION")

    if anomalies > 0:
        col_l, col_r = st.columns(2)
        with col_l:
            bucket = st.selectbox("BUCKET", ["HOUR", "DAY", "WEEK"], index=1)
            freq = {"HOUR": "h", "DAY": "D", "WEEK": "W"}[bucket]
            bucketed = df[df["is_anomaly"] == 1].set_index("timestamp").resample(freq).size().reset_index(name="count")
            dfig = go.Figure(go.Bar(x=bucketed["timestamp"], y=bucketed["count"], marker_color=RED, opacity=0.75))
            dfig.update_layout(**retro_layout(280, margin=dict(l=40, r=20, t=10, b=30), yaxis=dict(title=f"PER {bucket}")))
            st.plotly_chart(dfig, width="stretch")

        with col_r:
            cum = df["is_anomaly"].cumsum()
            cfig = go.Figure(go.Scatter(x=df["timestamp"].values[::100], y=cum.values[::100],
                mode="lines", line=dict(color=RED, width=1.5),
                fill="tozeroy", fillcolor="rgba(255,51,51,0.04)"))
            cfig.update_layout(**retro_layout(280, margin=dict(l=40, r=20, t=10, b=30), yaxis=dict(title="CUMULATIVE")))
            st.plotly_chart(cfig, width="stretch")

        if n_clusters > 0:
            st.markdown("##### CLUSTER LENGTHS")
            clfig = go.Figure(go.Histogram(x=c_lengths, nbinsx=min(30, max(c_lengths)), marker_color=CYAN, opacity=0.75))
            clfig.update_layout(**retro_layout(250, margin=dict(l=40, r=20, t=10, b=30),
                xaxis=dict(title="LENGTH (POINTS)"), yaxis=dict(title="FREQUENCY")))
            st.plotly_chart(clfig, width="stretch")
    else:
        st.warning("NO ANOMALIES DETECTED")

# ── OBSERVATION LOG ──────────────────────────────────────────────────────────

with tab_log:
    st.markdown("#### OBSERVATION LOG")
    col_f, col_n = st.columns([3, 1])
    with col_f:
        view = st.radio("FILTER", ["ALL", "ANOMALIES", "NOMINAL"], horizontal=True, label_visibility="collapsed")

    filt = df if view == "ALL" else df[df["is_anomaly"] == (1 if view == "ANOMALIES" else 0)]
    col_n.caption(f"{int(len(filt)):,} RECORDS")

    display = filt[["id", "timestamp", "is_anomaly"]].copy()
    display["CLASS"]  = display["is_anomaly"].map({1: "ANOMALY", 0: "NOMINAL"})
    display["STATUS"] = display["is_anomaly"].map({1: "DETECTED", 0: "NOMINAL"})
    st.dataframe(display[["id", "timestamp", "CLASS", "STATUS"]].rename(
        columns={"id": "ID", "timestamp": "TIMESTAMP"}),
        width="stretch", height=500, hide_index=True)

st.divider()
st.caption("SENTINEL V1.0.0 // ESA SPACECRAFT TELEMETRY ANOMALY DETECTION // PCA RESIDUAL MODEL")