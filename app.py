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
        cursor.execute("SELECT price FROM products WHERE name = ?", (item_name.lower(),))
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

# 5. UI Layout - Define columns first
col_video, col_billing = st.columns([2, 1])

with col_video:
    st.subheader("Live Scanning")
    ctx = webrtc_streamer(key="billing", video_processor_factory=VideoProcessor)

with col_billing:
    st.subheader("Current Bill")
    bill_area = st.empty()  # Slot for the table
    total_area = st.empty() # Slot for the total price
    
    # Place the button HERE so it is always visible
    finalize_button = st.button("Finalize Sale", type="primary", use_container_width=True)

# 6. Live Update Logic
if ctx.state.playing:
    # Use a placeholder loop that won't crash the browser
    while True:
        if ctx.video_processor:
            with ctx.video_processor.lock:
                detected_list = ctx.video_processor.current_detections
            
            if detected_list:
                grand_total = 0.0
                table_content = "| Item | Qty | Price | Subtotal |\n| :--- | :--- | :--- | :--- |\n"
                
                valid_items_found = False
                for item_name in set(detected_list):
                    price = get_db_info(item_name)
                    if price:
                        valid_items_found = True
                        qty = detected_list.count(item_name)
                        subtotal = price * qty
                        grand_total += subtotal
                        table_content += f"| {item_name.capitalize()} | {qty} | ₹{price} | **₹{subtotal}** |\n"

                if valid_items_found:
                    bill_area.markdown(table_content)
                    total_area.markdown(f"## Total: ₹{grand_total:.2f}")
                else:
                    bill_area.warning(f"Detected {', '.join(set(detected_list))} but they are not in mall.db")
            else:
                bill_area.info("Scanning for items...")
                total_area.empty()
        
        # Check if the user clicked the button during the loop
        if finalize_button:
            st.balloons()
            st.success("Transaction Complete!")
            break # Exit loop on finalize
            
        time.sleep(1) # Slow down refresh to 1 second to keep UI responsive
