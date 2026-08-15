"""
ECG Trace Annotator - MVP
Upload -> Visualize -> Annotate -> Review -> Export

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from annotation_model import Annotation, AnnotationStore, DEFAULT_LABELS  # noqa: E402
from io_utils import load_csv, guess_time_column, infer_sampling_rate, summarize  # noqa: E402
from signal_processing import apply_pipeline  # noqa: E402

st.set_page_config(page_title="ECG Trace Annotator", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = AnnotationStore()
if "df" not in st.session_state:
    st.session_state.df = None
if "time_col" not in st.session_state:
    st.session_state.time_col = None
if "signal_cols" not in st.session_state:
    st.session_state.signal_cols = []
if "fs" not in st.session_state:
    st.session_state.fs = None
if "source_file" not in st.session_state:
    st.session_state.source_file = ""
if "pending_start" not in st.session_state:
    st.session_state.pending_start = None
if "pending_end" not in st.session_state:
    st.session_state.pending_end = None

store: AnnotationStore = st.session_state.store

# ---------------------------------------------------------------------------
# Sidebar: file upload + channel selection + filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ECG Trace Annotator")

    st.header("1. Upload")
    uploaded = st.file_uploader("ECG CSV file", type=["csv"])
    use_sample = st.button("Load sample data")

    if uploaded is not None:
        df = load_csv(uploaded.read())
        st.session_state.df = df
        st.session_state.source_file = uploaded.name
        guessed = guess_time_column(list(df.columns))
        st.session_state.time_col = guessed or df.columns[0]
        st.session_state.signal_cols = [c for c in df.columns if c != st.session_state.time_col]
        st.session_state.store.clear()

    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_ecg.csv")
        df = pd.read_csv(sample_path)
        st.session_state.df = df
        st.session_state.source_file = "sample_ecg.csv"
        st.session_state.time_col = "time"
        st.session_state.signal_cols = [c for c in df.columns if c != "time"]
        st.session_state.store.clear()

    df = st.session_state.df

    if df is not None:
        st.header("2. Columns")
        time_col = st.selectbox(
            "Time column", options=list(df.columns),
            index=list(df.columns).index(st.session_state.time_col),
        )
        st.session_state.time_col = time_col

        available_signal_cols = [c for c in df.columns if c != time_col]
        signal_cols = st.multiselect(
            "Signal channel(s)", options=available_signal_cols,
            default=[c for c in st.session_state.signal_cols if c in available_signal_cols] or available_signal_cols[:1],
        )
        st.session_state.signal_cols = signal_cols

        if signal_cols:
            fs = infer_sampling_rate(df[time_col].to_numpy(dtype=float))
            st.session_state.fs = fs
            info = summarize(df, time_col, signal_cols)
            st.caption(
                f"{info['n_samples']} samples · "
                f"{info['sampling_rate_hz'] or '?'} Hz · "
                f"{info['duration_s']}s duration"
            )

        st.header("3. Signal processing")
        st.caption("Filters only affect the display trace, never the raw data.")
        show_filtered = st.checkbox("Show filtered trace", value=False)
        baseline = st.checkbox("Remove baseline wander", value=False, disabled=not show_filtered)
        lowpass_on = st.checkbox("Low-pass filter", value=False, disabled=not show_filtered)
        lowpass_hz = st.number_input("Low-pass cutoff (Hz)", value=40.0, disabled=not (show_filtered and lowpass_on))
        highpass_on = st.checkbox("High-pass filter", value=False, disabled=not show_filtered)
        highpass_hz = st.number_input("High-pass cutoff (Hz)", value=0.5, disabled=not (show_filtered and highpass_on))
        notch_on = st.checkbox("Notch filter", value=False, disabled=not show_filtered)
        notch_hz = st.number_input("Notch frequency (Hz)", value=50.0, disabled=not (show_filtered and notch_on))

        st.header("4. Export")
        json_bytes = store.to_json(st.session_state.source_file).encode("utf-8")
        st.download_button("Download annotations (JSON)", data=json_bytes,
                            file_name="annotations.json", mime="application/json")

        csv_rows = store.to_csv_rows(st.session_state.source_file)
        csv_bytes = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8") if csv_rows else b""
        st.download_button("Download annotations (CSV)", data=csv_bytes,
                            file_name="annotations.csv", mime="text/csv",
                            disabled=not csv_rows)

        st.header("5. Import")
        ann_json = st.file_uploader("Load annotations.json", type=["json"], key="ann_upload")
        if ann_json is not None:
            if st.button("Apply imported annotations"):
                st.session_state.store = AnnotationStore.from_json(ann_json.read().decode("utf-8"))
                st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if df is None:
    st.info("Upload a CSV (time + one or more signal columns), or click **Load sample data** in the sidebar to try the tool.")
    st.stop()

time_col = st.session_state.time_col
signal_cols = st.session_state.signal_cols
fs = st.session_state.fs

if not signal_cols:
    st.warning("Select at least one signal channel in the sidebar.")
    st.stop()

col_plot, col_ann = st.columns([3, 2])

# ---- Waveform plot ---------------------------------------------------------
with col_plot:
    st.subheader("ECG Trace")

    time_vals = df[time_col].to_numpy(dtype=float)
    fig = go.Figure()

    for ch in signal_cols:
        y = df[ch].to_numpy(dtype=float)
        if show_filtered:
            y = apply_pipeline(
                y, fs,
                baseline=baseline,
                lowpass_hz=lowpass_hz if lowpass_on else None,
                highpass_hz=highpass_hz if highpass_on else None,
                notch_hz=notch_hz if notch_on else None,
            )
        fig.add_trace(go.Scatter(x=time_vals, y=y, mode="lines", name=ch, line=dict(width=1.3)))

    # Draw existing annotations as shaded regions / markers
    for a in store.items:
        if a.type == "interval":
            fig.add_vrect(x0=a.start, x1=a.end, fillcolor="orange", opacity=0.25,
                           line_width=0, annotation_text=a.label, annotation_position="top left")
        else:
            fig.add_vline(x=a.time, line_width=1.5, line_dash="dash", line_color="crimson",
                           annotation_text=a.label, annotation_position="top")

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        dragmode="select",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    event = st.plotly_chart(
        fig, use_container_width=True, key="ecg_chart",
        on_select="rerun", selection_mode=("box",),
    )

    # Capture box-select as a candidate interval annotation
    sel = event.get("selection") if event else None
    if sel and sel.get("box"):
        box = sel["box"][0]
        xr = box.get("x", [])
        if len(xr) == 2:
            st.session_state.pending_start = round(min(xr), 4)
            st.session_state.pending_end = round(max(xr), 4)

    st.caption("Drag a box on the plot to select a region for a new interval annotation, or use the manual fields below.")

# ---- Annotation creation + table ------------------------------------------
with col_ann:
    st.subheader("Add Annotation")

    ann_type = st.radio("Type", options=["interval", "point"], horizontal=True)
    label_choice = st.selectbox("Label", options=DEFAULT_LABELS)
    custom_label = st.text_input("Custom label (optional, overrides selection)")
    channel_for_ann = st.selectbox("Channel", options=signal_cols)

    label = custom_label.strip() if custom_label.strip() else label_choice

    if ann_type == "interval":
        default_start = st.session_state.pending_start if st.session_state.pending_start is not None else float(time_vals[0])
        default_end = st.session_state.pending_end if st.session_state.pending_end is not None else float(time_vals[0]) + 0.1
        c1, c2 = st.columns(2)
        start = c1.number_input("Start (s)", value=float(default_start), format="%.4f")
        end = c2.number_input("End (s)", value=float(default_end), format="%.4f")
    else:
        default_time = st.session_state.pending_start if st.session_state.pending_start is not None else float(time_vals[0])
        point_time = st.number_input("Time (s)", value=float(default_time), format="%.4f")

    confidence = st.slider("Confidence", 0.0, 1.0, 1.0, 0.01)
    notes = st.text_input("Notes")

    if st.button("Add annotation", type="primary"):
        try:
            if ann_type == "interval":
                ann = Annotation(label=label, type="interval", channel=channel_for_ann,
                                  start=start, end=end, confidence=confidence, notes=notes)
            else:
                ann = Annotation(label=label, type="point", channel=channel_for_ann,
                                  time=point_time, confidence=confidence, notes=notes)
            store.add(ann)
            st.session_state.pending_start = None
            st.session_state.pending_end = None
            st.success(f"Added {label}")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    st.divider()
    st.subheader(f"Annotations ({len(store.items)})")

    if not store.items:
        st.caption("No annotations yet.")
    else:
        for a in store.items:
            loc = f"{a.start:.3f}–{a.end:.3f}s" if a.type == "interval" else f"{a.time:.3f}s"
            with st.expander(f"{a.label}  ·  {loc}  ·  {a.channel}"):
                st.write(f"**Type:** {a.type}  |  **Confidence:** {a.confidence:.2f}")
                if a.notes:
                    st.write(f"**Notes:** {a.notes}")
                b1, b2 = st.columns(2)
                if b1.button("Duplicate", key=f"dup_{a.id}"):
                    store.duplicate(a.id)
                    st.rerun()
                if b2.button("Delete", key=f"del_{a.id}"):
                    store.delete(a.id)
                    st.rerun()

st.divider()
with st.expander("Raw annotation JSON"):
    st.code(store.to_json(st.session_state.source_file), language="json")
