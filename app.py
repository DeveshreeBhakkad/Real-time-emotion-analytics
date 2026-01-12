
# Real-Time Emotion Analytics Dashboard 


import cv2
import streamlit as st
from deepface import DeepFace
from collections import deque, Counter
import pandas as pd
import time

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Emotion Analytics Dashboard",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #eaeaea;
}

.dashboard-title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 2px;
}
.dashboard-subtitle {
    font-size: 13px;
    color: #9aa0a6;
    margin-bottom: 18px;
}

/* Cards */
.card {
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 14px;
    background-color: #161b22;
}

/* Metric text */
.metric-label {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 6px;
}

/* Progress bar */
.progress-bg {
    width: 100%;
    height: 10px;
    background-color: #ffffff;
    border-radius: 6px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    text-align: right;
    padding-right: 6px;
    font-size: 10px;
    font-weight: 600;
    color: #ffffff;
    line-height: 10px;
}

/* Colors */
.green { background-color: #2ea043; }
.blue { background-color: #1f6feb; }
.orange { background-color: #d29922; }
.red { background-color: #f85149; }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------

st.markdown("<div class='dashboard-title'>Real-Time Emotion Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='dashboard-subtitle'>Live facial emotion recognition with session-based insights</div>", unsafe_allow_html=True)

# -------------------- FACE DETECTOR --------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------- SESSION STATE --------------------

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=10)

if "session_counts" not in st.session_state:
    st.session_state.session_counts = Counter()

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# -------------------- FUNCTIONS --------------------

def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        return result["dominant_emotion"]
    except:
        return "Unknown"

def progress_card(label, value, percent, color):
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="progress-bg">
            <div class="progress-fill {color}" style="width:{percent}%;">
                {int(percent)}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- LAYOUT --------------------

left, right = st.columns([1, 2], gap="large")

# ================= LEFT PANEL =================

with left:

    # Controls
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if st.button("🟢 Start Detection", use_container_width=True):
        st.session_state.camera_running = True
        st.session_state.start_time = time.time()

    if st.button("🔴 Stop Detection", use_container_width=True):
        st.session_state.camera_running = False

    status = "Running" if st.session_state.camera_running else "Stopped"
    st.markdown(f"**Status:** `{status}`")

    st.markdown("</div>", unsafe_allow_html=True)

    # Metrics
    faces = sum(st.session_state.session_counts.values())
    frames = st.session_state.frame_count

    max_ref = max(faces, frames, 1)

    dominant = (
        st.session_state.session_counts.most_common(1)[0][0]
        if st.session_state.session_counts else "N/A"
    )

    progress_card(
        "Faces Analyzed",
        faces,
        min((faces / max_ref) * 100, 100),
        "green"
    )

    progress_card(
        "Frames Processed",
        frames,
        min((frames / max_ref) * 100, 100),
        "blue"
    )

    progress_card(
        "Overall Mood",
        dominant,
        100,
        "orange"
    )

# ================= RIGHT PANEL =================

with right:
    video_box = st.empty()
    chart_box = st.empty()

# -------------------- CAMERA LOOP --------------------

if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces_detected:
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

# -------------------- FOOTER --------------------

st.markdown("---")
st.caption("Emotion Analytics Dashboard • AIML Project • Built by Deveshree")
