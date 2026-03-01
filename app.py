import streamlit as st
import cv2
import sqlite3
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
import time

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

# 3. AI Model Loading
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 4. WebRTC Video Processor
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.current_detections = []
        self.lock = threading.Lock()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=0.5)
        
        names = [self.model.names[int(c)] for c in results[0].boxes.cls]
        
        with self.lock:
            self.current_detections = names

        return results[0].plot()

# 5. UI Layout
col_video, col_billing = st.columns([2, 1])

with col_video:
    st.subheader("Live Scanning")
    ctx = webrtc_streamer(key="billing", video_processor_factory=VideoProcessor)

with col_billing:
    st.subheader("Current Bill")
    bill_area = st.empty()  # This placeholder will hold the table
    total_area = st.empty() # This placeholder will hold the sum

# 6. The "Live Update" Loop
# This part is CRITICAL: It keeps the UI alive to show the bill
while ctx.state.playing:
    if ctx.video_processor:
        with ctx.video_processor.lock:
            detected_list = ctx.video_processor.current_detections
        
        if detected_list:
            bill_items = []
            grand_total = 0.0
            
            # Header for Markdown Table
            table_content = "| Item | Qty | Price | Subtotal |\n| :--- | :--- | :--- | :--- |\n"
            
            for item_name in set(detected_list):
                price = get_db_info(item_name)
                if price:
                    qty = detected_list.count(item_name)
                    subtotal = price * qty
                    grand_total += subtotal
                    table_content += f"| {item_name.capitalize()} | {qty} | ₹{price} | **₹{subtotal}** |\n"

            bill_area.markdown(table_content)
            total_area.markdown(f"## Total: ₹{grand_total:.2f}")
        else:
            bill_area.info("Scanning for items...")
            total_area.empty()
    
    time.sleep(0.5) # Refresh the bill every half-second to save CPU
