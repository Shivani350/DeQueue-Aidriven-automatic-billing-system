import streamlit as st
import cv2
import sqlite3
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Page Setup
st.set_page_config(page_title="AI Smart Checkout Live", layout="wide")
st.title("🛒 AI Live Smart Checkout")

# 2. Database Function
def get_db_info(item_name):
    try:
        conn = sqlite3.connect('mall.db')
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM products WHERE name = ?", (item_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except:
        return None

# 3. AI Model Loading (Cached)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 4. WebRTC Video Processor
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = self.model(img, conf=0.5)
        annotated_img = results[0].plot()

        # Extract names for the sidebar/cart logic
        detected_names = [self.model.names[int(c)] for c in results[0].boxes.cls]
        
        # Note: In WebRTC, we return the image to the browser
        return annotated_img

# 5. UI Layout
col_video, col_billing = st.columns([2, 1])

with col_video:
    st.subheader("Live WebRTC Feed")
    webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

with col_billing:
    st.subheader("Real-time Detection")
    st.info("Items detected in the live feed will appear here. Note: WebRTC runs frame-by-frame; ensure a stable connection.")
    # For a full MVP, you would use a placeholder here to update the bill
