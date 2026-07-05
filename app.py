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

st.markdown("""
<style>
:root {
  --accent: #35d07f;
}
.stApp { background: #0a0e14; color: #e7edf5; }
h1, h2, h3 { color: #e7edf5 !important; }
.stMarkdown p { color: #90a0b7; }
div[data-testid="stMetricValue"] { color: #35d07f; }
.badge {
  display: inline-block; padding: 4px 12px; border-radius: 16px; font-weight: 700;
  font-size: 0.8rem; margin-right: 6px;
}
.badge-green { background: #1a3d2c; color: #35d07f; border: 1px solid #205036; }
.badge-yellow { background: #332405; color: #f2b544; border: 1px solid #5c460f; }
.badge-red { background: #331414; color: #ff6b57; border: 1px solid #5c1a1a; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Player Tracking - Analyze a Match")
st.markdown(
    "Upload a clip, click **Analyze**, get a full tracking report: player detection, "
    "team clustering, heatmaps, role clustering, and the original off-ball space-creation "
    "score - the same pipeline behind the "
    "[public demo dashboard](https://emanuele-clc.github.io/player-tracking-movement-analysis/), "
    "run on **your own video**."
)

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

st.divider()

uploaded = st.file_uploader("Upload a match clip", type=["mp4", "mov", "avi", "mkv"])

col_a, col_b = st.columns([1, 3])
with col_a:
    default_clip_id = _sanitize(uploaded.name) if uploaded else ""
    clip_id_input = st.text_input("Analysis name", value=default_clip_id, placeholder="e.g. friendly_vs_united")
with col_b:
    st.write("")
    st.write("")
    analyze_clicked = st.button("▶ Analyze", type="primary", disabled=uploaded is None)

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
    st.divider()
    st.header("Report")

    calib = result["calibration"]
    mode = calib.get("mode", "pixel")
    confidence = calib.get("confidence", 0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames analyzed", result["n_frames"])
    m2.metric("Tracks (players/ball/ref)", result["n_tracks"])
    m3.metric("Detections", result["n_detections"])
    m4.metric("Calibration confidence", f"{confidence:.0%}")

    if mode == "metric":
        st.markdown(
            f'<span class="badge badge-green">✓ Real-world calibration ({calib["status"]})</span>'
            f'<span class="badge badge-green">Metres &amp; speeds available</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-yellow">Pixel-space fallback</span>'
            '<span class="badge badge-yellow">No real metres/speeds this run</span>',
            unsafe_allow_html=True,
        )
    st.caption(calib.get("notes", ""))

    for w in result["warnings"]:
        st.warning(w)

    annotated_path = result.get("annotated_video")
    if annotated_path and Path(annotated_path).exists():
        st.subheader("Tracked video")
        st.video(annotated_path)

    teams_path = PROCESSED_DIR / "teams" / f"{clip_id}.parquet"
    if teams_path.exists():
        st.subheader("Team classification")
        st.dataframe(pd.read_parquet(teams_path), use_container_width=True)

    if mode == "metric":
        st.subheader("Heatmaps")
        cols = st.columns(3)
        labels = [("all", "All players"), ("team_0", "Team 0"), ("team_1", "Team 1")]
        for col, (key, label) in zip(cols, labels):
            img_path = PLOTS_DIR / f"heatmap_{key}_{clip_id}.png"
            if img_path.exists():
                col.image(str(img_path), caption=label, use_container_width=True)

    roles_path = PROCESSED_DIR / "role_clusters.parquet"
    if roles_path.exists():
        roles_df = pd.read_parquet(roles_path)
        roles_df = roles_df[roles_df["clip_id"] == clip_id]
        if not roles_df.empty:
            st.subheader("Role clustering")
            st.dataframe(roles_df.sort_values("role_cluster"), use_container_width=True)

    if mode == "metric":
        scores_path = PROCESSED_DIR / "space_creation_scores.parquet"
        if scores_path.exists():
            scores_df = pd.read_parquet(scores_path)
            scores_df = scores_df[scores_df["clip_id"] == clip_id]
            if not scores_df.empty:
                st.subheader("Off-ball space-creation score")
                st.dataframe(
                    scores_df.sort_values("space_creation_score", ascending=False),
                    use_container_width=True,
                )

    st.divider()
    dl_col, pub_col = st.columns(2)
    with dl_col:
        if TEMPLATE_PATH.exists():
            report_data = build_dashboard_data(clip_filter=[clip_id])
            report_html = render_html(report_data, TEMPLATE_PATH)
            st.download_button(
                "⬇ Download full interactive report (HTML)",
                data=report_html, file_name=f"{clip_id}_report.html", mime="text/html",
                use_container_width=True,
            )
    with pub_col:
        publish_clicked = st.button(
            "🚀 Publish to public dashboard", use_container_width=True,
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
                    st.write(f"✅ {label}")
                else:
                    all_ok = False
                    st.write(f"❌ {label}")
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
