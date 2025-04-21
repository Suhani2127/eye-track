import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="AI Eye & Screen Distance Monitor", layout="centered")
st.title("👁️ AI Eye & Screen Distance Monitor")
st.markdown("Tracks blink rate and face-screen distance to protect your eyes 👨‍💻")

# Constants
EAR_THRESHOLD = 0.22
NO_BLINK_LIMIT = 5  # seconds
CLOSE_DISTANCE_THRESHOLD = 0.3  # normalized value

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh

# Blink Detection
def calculate_EAR(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    vertical1 = np.linalg.norm(p2 - p4)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

class EyeScreenProcessor(VideoProcessorBase):
    def __init__(self):
        self.blinks = 0
        self.last_blink_time = time.time()
        self.blink_alert = ""
        self.distance_alert = ""
        self.face_distance = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            result = face_mesh.process(img_rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                # Eye EAR
                left_eye = [362, 385, 387, 263, 373, 380]
                right_eye = [33, 160, 158, 133, 153, 144]
                left_ear = calculate_EAR(landmarks, left_eye)
                right_ear = calculate_EAR(landmarks, right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                # Blink logic
                if avg_ear < EAR_THRESHOLD:
                    self.last_blink_time = time.time()
                    self.blinks += 1
                    self.blink_alert = ""
                elif time.time() - self.last_blink_time > NO_BLINK_LIMIT:
                    self.blink_alert = "⚠️ You're not blinking enough!"

                # Face distance (estimate from eyes)
                eye_left = landmarks[33]
                eye_right = landmarks[263]
                self.face_distance = np.linalg.norm([eye_left.x - eye_right.x, eye_left.y - eye_right.y])

                if self.face_distance > CLOSE_DISTANCE_THRESHOLD:
                    self.distance_alert = "🛑 You're too close to the screen!"
                else:
                    self.distance_alert = ""

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit Webcam
ctx = webrtc_streamer(
    key="eye_screen_monitor",
    video_processor_factory=EyeScreenProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# UI Feedback
if ctx.video_processor:
    st.metric("Blink Count", ctx.video_processor.blinks)
    st.metric("Estimated Face Distance", round(ctx.video_processor.face_distance, 2))

    if ctx.video_processor.blink_alert:
        st.warning(ctx.video_processor.blink_alert)

    if ctx.video_processor.distance_alert:
        st.error(ctx.video_processor.distance_alert)

st.caption("💡 Tip: Follow 20-20-20 rule — every 20 mins, look 20ft away for 20s.")
