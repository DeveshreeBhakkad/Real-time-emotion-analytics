
# ============================================================
# Real-Time Emotion Analytics Dashboard
# Phase 2 UI – Product-Style Layout
# ============================================================

import cv2
import streamlit as st
from deepface import DeepFace
from collections import deque, Counter
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Real-Time Emotion Analytics",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #eaeaea;
}

/* Header */
.hero {
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 36px;
    font-weight: 700;
}
.hero-subtitle {
    font-size: 14px;
    color: #9aa0a6;
    margin-top: 6px;
}

/* Buttons */
.control-bar {
    display: flex;
    justify-content: center;
    gap: 14px;
    margin-bottom: 18px;
}
button {
    font-size: 13px !important;
    padding: 6px 14px !important;
}

/* Cards */
.card {
    background-color: #161b22;
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 14px;
}

/* Metrics */
.metric-label {
    font-size: 15px;
    font-weight: 600;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
}

/* Progress bar */
.progress-bg {
    width: 100%;
    height: 8px;
    background-color: #ffffff;
    border-radius: 6px;
    margin-top: 6px;
}
.progress-fill {
    height: 100%;
    border-radius: 6px;
}

/* Colors */
.green { background-color: #2ea043; }
.blue { background-color: #1f6feb; }
.orange { background-color: #d29922; }

/* Footer */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #0e1117;
    text-align: center;
    font-size: 12px;
    color: #9aa0a6;
    padding: 8px;
    border-top: 1px solid #222;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="hero-title">Real-Time Face Emotion Analytics</div>
    <div class="hero-subtitle">📊 Live facial emotion recognition with session insights</div>
</div>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=10)
if "session_counts" not in st.session_state:
    st.session_state.session_counts = Counter()
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# ---------------- CONTROLS ----------------

st.markdown("<div class='control-bar'>", unsafe_allow_html=True)
col_stop, col_start = st.columns(2)

with col_stop:
    if st.button("🔴 Stop"):
        st.session_state.camera_running = False

with col_start:
    if st.button("🟢 Start"):
        st.session_state.camera_running = True

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FACE DETECTOR ----------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(
            face_img,
            actions=["emotion"],
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        return result["dominant_emotion"]
    except:
        return "Unknown"

def metric_card(label, value, percent, color):
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="progress-bg">
            <div class="progress-fill {color}" style="width:{percent}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------

left, right = st.columns([1, 2], gap="large")

# LEFT PANEL (Metrics)
with left:
    faces = sum(st.session_state.session_counts.values())
    frames = st.session_state.frame_count
    dominant = (
        st.session_state.session_counts.most_common(1)[0][0]
        if st.session_state.session_counts else "N/A"
    )

    max_ref = max(faces, frames, 1)

    metric_card("Faces Analyzed", faces, min((faces/max_ref)*100, 100), "green")
    metric_card("Frames Processed", frames, min((frames/max_ref)*100, 100), "blue")
    metric_card("Overall Mood", dominant.capitalize(), 100, "orange")

# RIGHT PANEL (Video + Chart)
with right:
    video_box = st.empty()
    chart_box = st.empty()

# ---------------- CAMERA LOOP ----------------

if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion = analyze_emotion(face_img)
            st.session_state.emotion_history.append(emotion)
            stable = Counter(st.session_state.emotion_history).most_common(1)[0][0]
            st.session_state.session_counts[stable] += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,180,0), 2)
            cv2.putText(frame, stable, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        video_box.image(frame, channels="BGR")

        if st.session_state.session_counts:
            df = pd.DataFrame.from_dict(
                st.session_state.session_counts,
                orient="index",
                columns=["Count"]
            )
            chart_box.bar_chart(df)

    cap.release()

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
⚠️ Emotion recognition is probabilistic and may vary due to lighting,
camera quality, facial angle, and individual differences.
</div>
""", unsafe_allow_html=True)
