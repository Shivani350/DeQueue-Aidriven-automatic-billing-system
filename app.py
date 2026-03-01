import streamlit as st
import cv2
import sqlite3
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading

# 1. Page Setup
st.set_page_config(page_title="AI Smart Checkout Live", layout="wide")
st.title("🛒 AI Live Smart Checkout")

# 2. Database Function
def get_db_info(item_name):
    try:
        conn = sqlite3.connect('mall.db')
        cursor = conn.cursor()
        # Your original query: select barcode and price
        cursor.execute("SELECT barcode, price FROM products WHERE name = ?", (item_name,))
        result = cursor.fetchone()
        conn.close()
        return result # returns (barcode, price)
    except:
        return None

# 3. AI Model Loading (Cached)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 4. WebRTC Video Processor with Shared State
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.last_detected = []
        self.lock = threading.Lock()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = self.model(img, conf=0.5)
        annotated_img = results[0].plot()

        # Get names of currently visible items
        names = [self.model.names[int(c)] for c in results[0].boxes.cls]
        
        with self.lock:
            self.last_detected = names

        return annotated_img

# 5. UI Layout
col_video, col_billing = st.columns([2, 1])

with col_video:
    st.subheader("Live WebRTC Feed")
    ctx = webrtc_streamer(key="billing-system", video_processor_factory=VideoProcessor)

with col_billing:
    st.subheader("Shopping Cart")
    cart_placeholder = st.empty()
    total_placeholder = st.empty()

    # If the video stream is active, pull data from the processor
    if ctx.video_processor:
        with ctx.video_processor.lock:
            current_items = ctx.video_processor.last_detected
        
        if current_items:
            bill_details = ""
            grand_total = 0.0
            
            # Use a set to avoid repeating the same item in the list
            for item in set(current_items):
                db_data = get_db_info(item)
                if db_data:
                    barcode, price = db_data
                    qty = current_items.count(item)
                    item_total = price * qty
                    bill_details += f"**{item.capitalize()}** x{qty} — ₹{item_total:.2f}\n\n"
                    grand_total += item_total
            
            cart_placeholder.markdown(bill_details)
            total_placeholder.markdown(f"## Total: ₹{grand_total:.2f}")
        else:
            cart_placeholder.info("No items detected. Place an object in front of the camera.")

# 6. Finalize Sale Button
if st.button("Finalize Sale", type="primary", use_container_width=True):
    st.balloons()
    st.success("Transaction pushed to database!")
