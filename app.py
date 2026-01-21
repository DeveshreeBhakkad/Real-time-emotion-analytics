import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real-Time Face Emotion Analytics",
    layout="wide"
)

# ---------------- ENV CHECK ----------------
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"

# ---------------- SESSION STATE ----------------
if "running" not in st.session_state:
    st.session_state.running = False
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "frames" not in st.session_state:
    st.session_state.frames = 0

# ---------------- FIXED FOOTER STYLE ----------------
st.markdown(
    """
    <style>
    .footer-warning {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #020617;
        color: #facc15;
        text-align: center;
        padding: 10px;
        font-weight: 600;
        border-top: 1px solid #1e293b;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style="text-align:center; margin-top:-10px;">
        <h1>Real-Time Face Emotion Analytics</h1>
        <p>📊 Live facial emotion recognition with session-based insights</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- START / STOP ----------------
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([4, 1, 1, 1])

with btn_col4:
    start = st.button("🟢 Start")
    stop = st.button("🔴 Stop")

if start:
    st.session_state.running = True
    st.session_state.emotion_log = []
    st.session_state.frames = 0

if stop:
    st.session_state.running = False

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
left_col, center_col, right_col = st.columns([1.2, 2.2, 2])

# ---------------- CLOUD MESSAGE ----------------
if IS_CLOUD:
    center_col.info(
        "🔒 Live webcam access is disabled on Streamlit Cloud.\n\n"
        "Run this app locally to enable real-time emotion detection."
    )

# ---------------- LIVE CAMERA (LOCAL ONLY) ----------------
if st.session_state.running and not IS_CLOUD:
    import cv2
    from deepface import DeepFace  # LAZY IMPORT (KEY FIX)

    cap = cv2.VideoCapture(0)
    cam_box = center_col.empty()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frames += 1

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            emotion = result[0]["dominant_emotion"]
            st.session_state.emotion_log.append(emotion)

            face = result[0]["region"]
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        except:
            pass

        cam_box.image(frame, channels="BGR")

    cap.release()

# ---------------- METRICS ----------------
with left_col:
    st.markdown("### Faces Analyzed")
    st.markdown(f"## {len(st.session_state.emotion_log)}")
    st.progress(min(len(st.session_state.emotion_log) / 100, 1.0))

    st.markdown("### Frames Processed")
    st.markdown(f"## {st.session_state.frames}")
    st.progress(min(st.session_state.frames / 100, 1.0))

    if st.session_state.emotion_log:
        mood = Counter(st.session_state.emotion_log).most_common(1)[0][0]
        st.markdown("### Overall Mood")
        st.markdown(f"## {mood.capitalize()}")

# ---------------- SESSION SUMMARY ----------------
if not st.session_state.running and st.session_state.emotion_log:
    counts = Counter(st.session_state.emotion_log)
    df = pd.DataFrame(counts.items(), columns=["Emotion", "Count"])
    df["Percentage (%)"] = (df["Count"] / df["Count"].sum() * 100).round(1)

    with center_col:
        st.markdown("## 📊 Session Summary")
        fig, ax = plt.subplots()
        ax.bar(df["Emotion"], df["Count"])
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with right_col:
        st.markdown("## Emotion Percentage")
        st.dataframe(df, use_container_width=True)

        fig2, ax2 = plt.subplots()
        ax2.pie(
            df["Count"],
            labels=df["Emotion"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.axis("equal")
        st.pyplot(fig2)

# ---------------- FIXED FOOTER ----------------
st.markdown(
    """
    <div class="footer-warning">
        ⚠️ Emotion recognition is probabilistic and may vary due to lighting,
        camera quality, facial angle, and individual differences.
    </div>
    """,
    unsafe_allow_html=True
)
