import cv2
import time
import streamlit as st
import pandas as pd
from collections import Counter
from deepface import DeepFace

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Real-Time Face Emotion Analytics",
    layout="wide"
)

# ================= LOAD CSS =================
with open("ui/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ================= HEADER (FIXED + CENTERED) =================
st.markdown("""
<div class="header">
  <h1>Real-Time Face Emotion Analytics</h1>
  <p>📊 Live facial emotion recognition with session-based insights</p>
</div>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "running" not in st.session_state:
    st.session_state.running = False
if "emotions" not in st.session_state:
    st.session_state.emotions = []
if "frames" not in st.session_state:
    st.session_state.frames = 0

# ================= FIXED CONTROLS =================
st.markdown("""
<div class="controls">
""", unsafe_allow_html=True)

start = st.button("🟢 Start")
stop = st.button("🔴 Stop")

st.markdown("</div>", unsafe_allow_html=True)

if start:
    st.session_state.running = True
    st.session_state.emotions = []
    st.session_state.frames = 0

if stop:
    st.session_state.running = False

# ================= MAIN LAYOUT =================
col_cards, col_video, col_summary = st.columns([1, 1.5, 1.3], gap="medium")

# ================= METRIC CARDS =================
with col_cards:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Faces Analyzed")
    faces = len(st.session_state.emotions)
    st.markdown(f"### {faces}")
    st.progress(min(faces / 50, 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Frames Processed")
    st.markdown(f"### {st.session_state.frames}")
    st.progress(min(st.session_state.frames / 50, 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    mood = Counter(st.session_state.emotions).most_common(1)[0][0] if st.session_state.emotions else "—"
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Overall Mood")
    st.markdown(f"### {mood.capitalize()}")
    st.progress(1.0)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= VIDEO FEED =================
with col_video:
    st.subheader("Live Camera Feed")
    video_box = st.empty()

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            st.session_state.frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]

                try:
                    result = DeepFace.analyze(
                        face_img,
                        actions=["emotion"],
                        enforce_detection=False
                    )
                    emotion = result[0]["dominant_emotion"]
                    st.session_state.emotions.append(emotion)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_box.image(frame, channels="RGB")
            time.sleep(0.05)

        cap.release()

# ================= SESSION SUMMARY =================
with col_summary:
    if not st.session_state.running and st.session_state.emotions:
        st.subheader("📊 Session Summary")
        counts = Counter(st.session_state.emotions)
        df = pd.DataFrame(counts.items(), columns=["Emotion", "Count"]).set_index("Emotion")
        st.bar_chart(df)

# ================= FOOTER =================
st.markdown("""
<div class="footer-warning">
⚠️ Emotion recognition is probabilistic and may vary due to lighting,
camera quality, facial angle, and individual differences.
</div>
""", unsafe_allow_html=True)
