"""
Local demo app: upload a football clip, click Analyze, get a full tracking
report - the "pitch this to a club" experience for the whole pipeline.

This is deliberately a LOCAL app, not something hosted on GitHub Pages:
GitHub Pages only serves static files, it cannot run the YOLO detection /
tracking / calibration code this analysis needs. Running it locally means no
hosting cost and the uploaded video never leaves your machine.

Run it with:
    pip install -r requirements.txt
    pip install streamlit
    streamlit run app.py

Then open the local URL Streamlit prints (usually http://localhost:8501),
upload a clip, and click Analyze.
"""
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import run_pipeline  # noqa: E402
from generate_dashboard_data import build_dashboard_data, render_html  # noqa: E402

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "plots"
TEMPLATE_PATH = ROOT / "docs" / "_index_template.html"

st.set_page_config(page_title="Player Tracking - Analyze a match", layout="wide", page_icon="⚽")

# ---------------------------------------------------------------------------
# Theme: matches the public dashboard (docs/_index_template.html) so the local
# app and the live site feel like the same product, not two different demos.
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --accent: #35d07f;
  --accent-2: #4fa8ff;
  --bg: #0a0e14;
  --panel: #121722;
  --panel-2: #161d2b;
  --border: #232d3f;
  --text: #e7edf5;
  --muted: #90a0b7;
  --muted-2: #5f7089;
  --team0: #4fa8ff;
  --team1: #ff6b57;
  --ref: #f2c744;
  --yellow: #f2b544;
  --red: #ff6b57;
}
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 800 !important; letter-spacing: -0.01em; }
.stMarkdown p, .stMarkdown li, label, .stCaption { color: var(--muted); }
hr { border-color: var(--border) !important; }

/* Hero header */
.app-hero {
  background: linear-gradient(135deg, #10321f 0%, #0a0e14 60%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 22px;
}
.app-hero h1 { margin: 0 0 8px; font-size: 1.9rem; }
.app-hero p { margin: 0; color: var(--muted); font-size: 0.98rem; max-width: 780px; }
.how-steps { display: flex; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
.how-step {
  flex: 1; min-width: 160px; background: rgba(255,255,255,0.03);
  border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
}
.how-step .n {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%; background: var(--accent);
  color: #06130c; font-weight: 800; font-size: 0.78rem; margin-right: 8px;
}
.how-step span.label { color: var(--text); font-weight: 600; font-size: 0.88rem; }

/* Section panels - wraps st.container(border=True) */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel); border-color: var(--border) !important;
  border-radius: 14px !important;
}
.section-eyebrow {
  color: var(--accent); font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 2px;
}
.section-title { font-size: 1.15rem; font-weight: 800; color: var(--text); margin: 0 0 6px; }
.section-explain { color: var(--muted); font-size: 0.87rem; margin-bottom: 14px; line-height: 1.5; }

/* Insights callout */
.insights-box {
  background: var(--panel-2); border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: 10px; padding: 16px 20px; margin-bottom: 4px;
}
.insights-box ul { margin: 6px 0 0; padding-left: 18px; }
.insights-box li { color: var(--text); font-size: 0.92rem; line-height: 1.8; }

/* Badges */
.badge {
  display: inline-block; padding: 5px 14px; border-radius: 16px; font-weight: 700;
  font-size: 0.8rem; margin-right: 6px; margin-bottom: 6px;
}
.badge-green { background: #1a3d2c; color: var(--accent); border: 1px solid #205036; }
.badge-yellow { background: #332405; color: var(--yellow); border: 1px solid #5c460f; }
.badge-red { background: #331414; color: var(--red); border: 1px solid #5c1a1a; }

/* Metrics */
div[data-testid="stMetric"] {
  background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px;
}
div[data-testid="stMetricValue"] { color: var(--accent); font-weight: 800; }
div[data-testid="stMetricLabel"] { color: var(--muted); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 8px; font-weight: 700; border: 1px solid var(--border);
}
.stButton > button[kind="primary"] { background: var(--accent); color: #06130c; border: none; }

/* Dataframes */
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* File uploader */
div[data-testid="stFileUploaderDropzone"] {
  background: var(--panel-2); border: 1.5px dashed var(--border); border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-hero">
  <h1>⚽ Player Tracking &mdash; Analyze a Match</h1>
  <p>Upload a football clip and get a full tracking report back: every player detected and followed
  frame by frame, split into teams, turned into heatmaps, movement roles, and the original off-ball
  space-creation score &mdash; the same engine behind the
  <a href="https://emanuele-clc.github.io/player-tracking-movement-analysis/" target="_blank" style="color:var(--accent-2)">public demo dashboard</a>,
  running here on <b>your own footage</b>, saved only on this machine.</p>
  <div class="how-steps">
    <div class="how-step"><span class="n">1</span><span class="label">Upload a clip below</span></div>
    <div class="how-step"><span class="n">2</span><span class="label">Click Analyze &amp; wait for the progress bar</span></div>
    <div class="how-step"><span class="n">3</span><span class="label">Read the report, download it, or publish it live</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "clip_id" not in st.session_state:
    st.session_state.clip_id = None


def _sanitize(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(name).stem).strip("_").lower()
    return base or "clip"


def _run_git(args):
    """Run a git command in the repo root, returning (ok, output). Never
    raises - git/network problems are shown to the user, not crashed on."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output.strip()
    except FileNotFoundError:
        return False, "git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return False, "git command timed out (check your network / GitHub credentials)."


def publish_clip_to_dashboard(clip_id: str):
    """Fold this clip into the public dashboard (docs/index.html) alongside
    every other clip already analyzed, then commit and push - so the site
    updates automatically from this same local app, no manual git commands.
    Only works if this checkout has push access to its 'origin' remote
    (i.e. you're running this on your own machine, logged into your own
    GitHub account - exactly the normal way to run this app)."""
    steps = []

    data = build_dashboard_data(clip_filter=None)  # every clip, not just this one
    dashboard_json_path = ROOT / "docs" / "assets" / "dashboard_data.json"
    dashboard_json_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_json_path.write_text(json.dumps(data))
    if TEMPLATE_PATH.exists():
        html = render_html(data, TEMPLATE_PATH)
        (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
    steps.append(("Regenerated docs/index.html with every analyzed clip", True, ""))

    ok, out = _run_git(["add", "-A"])
    steps.append(("git add -A", ok, out))
    if not ok:
        return steps

    ok, out = _run_git(["commit", "-m", f"Add {clip_id} analysis to dashboard"])
    steps.append((f'git commit -m "Add {clip_id} analysis to dashboard"', ok, out))
    if not ok and "nothing to commit" not in out.lower():
        return steps

    ok, out = _run_git(["push"])
    steps.append(("git push", ok, out))
    return steps


def section_header(eyebrow, title, explain):
    st.markdown(
        f'<div class="section-eyebrow">{eyebrow}</div>'
        f'<div class="section-title">{title}</div>'
        f'<div class="section-explain">{explain}</div>',
        unsafe_allow_html=True,
    )


def fmt_teams_table(df):
    d = df.copy()
    rename = {"track_id": "Player ID", "team": "Team", "class": "Class"}
    d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
    keep = [c for c in ["Player ID", "Team", "Class"] if c in d.columns]
    return d[keep] if keep else d


def fmt_roles_table(df):
    d = df.copy()
    if "std_x" in d.columns and "std_y" in d.columns:
        d["Positional spread (m)"] = (d["std_x"] ** 2 + d["std_y"] ** 2) ** 0.5
    rename = {
        "track_id": "Player ID", "team": "Team", "n_frames": "Frames tracked",
        "mean_x": "Avg X (m)", "mean_y": "Avg Y (m)", "mean_speed": "Avg speed (m/s)",
        "role_cluster": "Movement cluster",
    }
    d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
    for c in ["Avg X (m)", "Avg Y (m)", "Positional spread (m)", "Avg speed (m/s)"]:
        if c in d.columns:
            d[c] = d[c].round(2)
    keep = [c for c in ["Player ID", "Team", "Frames tracked", "Avg X (m)", "Avg Y (m)",
                        "Positional spread (m)", "Avg speed (m/s)", "Movement cluster"] if c in d.columns]
    return d[keep] if keep else d


def fmt_scores_table(df):
    d = df.copy()
    rename = {
        "track_id": "Player ID", "n_valid_frames": "Valid frames",
        "space_creation_score": "Space-creation score (m2/m)",
    }
    d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
    if "Space-creation score (m2/m)" in d.columns:
        d["Space-creation score (m2/m)"] = d["Space-creation score (m2/m)"].round(1)
    keep = [c for c in ["Player ID", "Valid frames", "Space-creation score (m2/m)"] if c in d.columns]
    return d[keep] if keep else d


def compute_insights(mode, teams_df, roles_df, scores_df):
    insights = []
    if teams_df is not None and not teams_df.empty and "team" in teams_df.columns:
        counts = teams_df["team"].value_counts()
        if len(counts) >= 2:
            insights.append(
                f"<b>{counts.index[0]}</b> has the most classified players this clip "
                f"({int(counts.iloc[0])} of {int(counts.sum())})."
            )
    if mode == "metric" and roles_df is not None and not roles_df.empty and "mean_speed" in roles_df.columns:
        fastest = roles_df.sort_values("mean_speed", ascending=False).iloc[0]
        insights.append(
            f"Player <b>#{int(fastest['track_id'])}</b> ({fastest.get('team', 'unknown')}) covered the most "
            f"ground on average &mdash; {fastest['mean_speed']:.1f} m/s mean speed."
        )
    if mode == "metric" and scores_df is not None and not scores_df.empty and "space_creation_score" in scores_df.columns:
        top = scores_df.sort_values("space_creation_score", ascending=False).iloc[0]
        insights.append(
            f"Player <b>#{int(top['track_id'])}</b> ranks highest on the off-ball space-creation score "
            f"&mdash; their movement opened the most room for teammates."
        )
    if not insights:
        insights.append(
            "Not enough data in this clip yet for automatic insights &mdash; try a longer clip, "
            "or turn off Quick preview for a fuller pass."
        )
    return insights


with st.sidebar:
    st.header("Options")
    quick_mode = st.toggle("Quick preview (recommended)", value=True,
                            help="Analyze only the first several seconds instead of the whole video - "
                                 "much faster, ideal for a live demo. Turn off to process the full clip.")
    max_frames = st.slider("Frames to analyze", 30, 500, 150, step=10, disabled=not quick_mode,
                            help="At ~24-30fps this is roughly max_frames/25 seconds of match action.")
    with st.expander("Advanced"):
        weights = st.text_input(
            "Detection weights", value="yolov8n.pt",
            help="Default: stock YOLOv8 (person/ball only). Point this at a football-fine-tuned "
                 "checkpoint (see data/README.md) for far better player/ball/referee detection.",
        )
        conf = st.slider("Detection confidence threshold", 0.05, 0.9, 0.25, step=0.05)

st.markdown('<div class="section-eyebrow">Step 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload your clip</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop an .mp4 / .mov / .avi / .mkv file here, or click to browse",
                             type=["mp4", "mov", "avi", "mkv"])

col_a, col_b = st.columns([1, 3])
with col_a:
    default_clip_id = _sanitize(uploaded.name) if uploaded else ""
    clip_id_input = st.text_input("Name this analysis", value=default_clip_id, placeholder="e.g. friendly_vs_united")
with col_b:
    st.write("")
    st.write("")
    analyze_clicked = st.button("Analyze", type="primary", disabled=uploaded is None,
                                 use_container_width=False)

if analyze_clicked and uploaded is not None:
    clip_id = _sanitize(clip_id_input) + "_" + uuid.uuid4().hex[:6]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    video_path = RAW_DIR / f"{clip_id}{Path(uploaded.name).suffix.lower()}"
    video_path.write_bytes(uploaded.getvalue())

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_cb(pct, msg):
        progress_bar.progress(min(pct, 1.0))
        status_text.info(msg)

    t0 = time.time()
    try:
        summary = run_pipeline(
            video_path, clip_id,
            weights=weights, max_frames=max_frames if quick_mode else None,
            conf=conf, annotate=True, progress_cb=progress_cb,
        )
        summary["elapsed_s"] = round(time.time() - t0, 1)
        st.session_state.result = summary
        st.session_state.clip_id = clip_id
        status_text.success(f"Done in {summary['elapsed_s']}s.")
    except Exception as e:
        status_text.error(f"Analysis failed: {e}")
        st.exception(e)

result = st.session_state.result
clip_id = st.session_state.clip_id

if result:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Step 3</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title" style="font-size:1.4rem;">Report: {clip_id}</div>', unsafe_allow_html=True)

    calib = result["calibration"]
    mode = calib.get("mode", "pixel")
    confidence = calib.get("confidence", 0.0)

    teams_path = PROCESSED_DIR / "teams" / f"{clip_id}.parquet"
    teams_df = pd.read_parquet(teams_path) if teams_path.exists() else None

    roles_path = PROCESSED_DIR / "role_clusters.parquet"
    roles_df = None
    if roles_path.exists():
        roles_all = pd.read_parquet(roles_path)
        roles_df = roles_all[roles_all["clip_id"] == clip_id]
        if roles_df.empty:
            roles_df = None

    scores_path = PROCESSED_DIR / "space_creation_scores.parquet"
    scores_df = None
    if scores_path.exists():
        scores_all = pd.read_parquet(scores_path)
        scores_df = scores_all[scores_all["clip_id"] == clip_id]
        if scores_df.empty:
            scores_df = None

    insights = compute_insights(mode, teams_df, roles_df, scores_df)
    st.markdown(
        '<div class="insights-box">'
        '<div class="section-eyebrow">Auto-generated insights</div>'
        '<ul>' + "".join(f"<li>{i}</li>" for i in insights) + '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        section_header(
            "Overview", "How much was actually tracked",
            "These four numbers come straight from the detection/tracking stage - no manual "
            "counting involved.",
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames analyzed", result["n_frames"])
        m2.metric("Tracks (players/ball/ref)", result["n_tracks"])
        m3.metric("Detections", result["n_detections"])
        m4.metric("Calibration confidence", f"{confidence:.0%}")

        st.markdown("<br>", unsafe_allow_html=True)
        if mode == "metric":
            st.markdown(
                f'<span class="badge badge-green">Real-world calibration ({calib.get("status", "ok")})</span>'
                f'<span class="badge badge-green">Metres &amp; speeds available</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge badge-yellow">Pixel-space fallback</span>'
                '<span class="badge badge-yellow">No real metres/speeds this run</span>',
                unsafe_allow_html=True,
            )
        st.caption(
            calib.get("notes", "")
            + (" This means positions below are true pitch metres." if mode == "metric" else
               " This means tracking and team clustering below are still real, but distances, speeds, "
               "heatmaps, and the space-creation score need real metres, so they're skipped rather than faked.")
        )

        for w in result["warnings"]:
            st.warning(w)

    st.markdown("<br>", unsafe_allow_html=True)

    annotated_path = result.get("annotated_video")
    if annotated_path and Path(annotated_path).exists():
        with st.container(border=True):
            section_header(
                "Your footage", "Tracked video",
                "Your uploaded clip with every detected player/ball boxed and tracked frame by frame - "
                "exactly what the model saw, nothing touched up by hand.",
            )
            st.video(annotated_path)
        st.markdown("<br>", unsafe_allow_html=True)

    if teams_df is not None:
        with st.container(border=True):
            section_header(
                "Team ID", "Team classification",
                "Every tracked player split into Team 0 / Team 1 / referee-or-neutral, purely from "
                "clustering jersey colour - no manual tagging.",
            )
            st.dataframe(fmt_teams_table(teams_df), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if mode == "metric":
        with st.container(border=True):
            section_header(
                "Movement analysis", "Heatmaps",
                "Where each group of players actually spent their time on the pitch, built from every "
                "tracked position in the clip.",
            )
            cols = st.columns(3)
            labels = [("all", "All players"), ("team_0", "Team 0"), ("team_1", "Team 1")]
            any_shown = False
            for col, (key, label) in zip(cols, labels):
                img_path = PLOTS_DIR / f"heatmap_{key}_{clip_id}.png"
                if img_path.exists():
                    col.image(str(img_path), caption=label, use_container_width=True)
                    any_shown = True
            if not any_shown:
                st.caption("No heatmaps generated for this clip yet.")
        st.markdown("<br>", unsafe_allow_html=True)

    if roles_df is not None:
        with st.container(border=True):
            section_header(
                "Movement analysis", "Role clustering",
                "Per-player k-means clustering on average position, how much ground they covered, and "
                "average speed - grouping players by how they actually moved, not by a lineup sheet. "
                + ("Speed is real m/s here." if mode == "metric" else
                   "Positions are schematic (pixel-space fallback), so 'speed' isn't meaningful this run."),
            )
            st.dataframe(
                fmt_roles_table(roles_df.sort_values("role_cluster")),
                use_container_width=True, hide_index=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    if mode == "metric" and scores_df is not None:
        with st.container(border=True):
            section_header(
                "Original contribution", "Off-ball space-creation score",
                "Not where a player stood, but what their movement did for their team: a Voronoi-based "
                "estimate of how much extra space a player's own run opened up for teammates over the "
                "next couple of seconds. Higher = more space created by moving, not just by having the ball.",
            )
            st.dataframe(
                fmt_scores_table(scores_df.sort_values("space_creation_score", ascending=False)),
                use_container_width=True, hide_index=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        section_header(
            "Share this analysis", "Download or publish",
            "Keep this report to yourself, or push it live to your public dashboard - your choice.",
        )
        dl_col, pub_col = st.columns(2)
        with dl_col:
            if TEMPLATE_PATH.exists():
                report_data = build_dashboard_data(clip_filter=[clip_id])
                report_html = render_html(report_data, TEMPLATE_PATH)
                st.download_button(
                    "Download full interactive report (HTML)",
                    data=report_html, file_name=f"{clip_id}_report.html", mime="text/html",
                    use_container_width=True,
                )
        with pub_col:
            publish_clicked = st.button(
                "Publish to public dashboard", use_container_width=True,
                help="Regenerates docs/index.html with this clip added, then runs "
                     "git add / commit / push - only works if this checkout can push "
                     "to its own GitHub remote.",
            )

        if publish_clicked:
            with st.status("Publishing...", expanded=True) as status_box:
                steps = publish_clip_to_dashboard(clip_id)
                all_ok = True
                for label, ok, out in steps:
                    if ok:
                        st.write(f"OK: {label}")
                    else:
                        all_ok = False
                        st.write(f"FAILED: {label}")
                    if out:
                        st.code(out, language="text")
                if all_ok:
                    status_box.update(label="Published - check the live site in a minute.", state="complete")
                else:
                    status_box.update(label="Publish failed at the step above - see the output for why.", state="error")

        st.caption(
            "This report and all intermediate data are saved locally under data/processed/ - "
            "nothing is uploaded anywhere unless you click Publish, which just runs the same "
            "git commands you'd type yourself. Re-run with more frames (turn off Quick preview) "
            "for a full analysis once you're happy with a quick look."
        )
