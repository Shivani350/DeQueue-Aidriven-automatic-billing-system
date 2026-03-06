import streamlit as st
import cv2
import sqlite3
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import threading
import time
from datetime import datetime

# 1. Page Setup
st.set_page_config(page_title="AI Smart Checkout Live", layout="wide")
st.title("🛒 AI Live Smart Checkout")

# RTC Configuration for Cloud Deployment (STUN servers allow the camera to work over the internet)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

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
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.current_detections = []
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=0.5)
        names = [self.model.names[int(c)] for c in results[0].boxes.cls]
        
        with self.lock:
            self.current_detections = names
            
        return results[0].plot()

# 5. The Professional Pop-up Function
@st.dialog("🧾 Official Receipt")
def show_receipt(data, total):
    st.balloons() 
    st.write(f"**VISTA AI MART**")
    st.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.divider()
    
    st.markdown("| Item | Qty | Subtotal |")
    st.markdown("| :--- | :--- | :--- |")
    for item in data:
        st.markdown(f"| {item['name'].capitalize()} | {item['qty']} | ₹{item['sub']:.2f} |")
    
    st.divider()
    st.markdown(f"### **GRAND TOTAL: ₹{total:.2f}**")
    st.write("Thank you for shopping!")
    
    if st.button("Close & New Sale"):
        st.rerun()

# 6. UI Layout
col_video, col_billing = st.columns([2, 1])

with col_video:
    st.subheader("Live Scanning")
    ctx = webrtc_streamer(
        key="billing_feed", 
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False}
    )

with col_billing:
    st.subheader("Current Bill")
    bill_area = st.empty()  
    total_area = st.empty() 
    finalize_button = st.button("Finalize Sale", type="primary", use_container_width=True)

# 7. Live Update Logic
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            with ctx.video_processor.lock:
                detected_list = ctx.video_processor.current_detections
            
            grand_total = 0.0
            receipt_items = []
            
            if detected_list:
                table_content = "| Item | Qty | Price | Subtotal |\n| :--- | :--- | :--- | :--- |\n"
                valid_items = False
                
                for item_name in set(detected_list):
                    price = get_db_info(item_name)
                    if price:
                        valid_items = True
                        qty = detected_list.count(item_name)
                        subtotal = price * qty
                        grand_total += subtotal
                        table_content += f"| {item_name.capitalize()} | {qty} | ₹{price} | **₹{subtotal}** |\n"
                        receipt_items.append({"name": item_name, "qty": qty, "sub": subtotal})

                if valid_items:
                    bill_area.markdown(table_content)
                    total_area.markdown(f"## Total: ₹{grand_total:.2f}")
                else:
                    bill_area.warning("Scanning items... (Update mall.db if prices missing)")
            else:
                bill_area.info("Bring an item to camera...")
                total_area.empty()

        # Handle Finalize Click
        if finalize_button:
            if grand_total > 0:
                show_receipt(receipt_items, grand_total)
                break 
            else:
                st.warning("No items detected to finalize!")
                break
                
        time.sleep(0.5)


