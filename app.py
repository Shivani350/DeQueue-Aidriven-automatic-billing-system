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

# 2. Database Function (Fetches individual price)
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

# 4. The "Brain" - WebRTC Video Processor
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.current_detections = []
        self.lock = threading.Lock()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=0.5)
        
        # Get every single object found in the frame (allows for duplicates)
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
    # Placeholders allow the text to refresh without moving the whole page
    cart_table = st.empty()
    total_display = st.empty()

    if ctx.video_processor:
        # This loop keeps the UI updated with the latest AI data
        with ctx.video_processor.lock:
            detected_list = ctx.video_processor.current_detections
        
        if detected_list:
            bill_items = []
            grand_total = 0.0
            
            # Count each unique item found
            for item_name in set(detected_list):
                price = get_db_info(item_name)
                if price:
                    qty = detected_list.count(item_name) # Counts if there are 2+ of the same item
                    subtotal = price * qty
                    grand_total += subtotal
                    bill_items.append(f"| {item_name.capitalize()} | {qty} | ₹{price} | **₹{subtotal}** |")

            # Update the table and total
            if bill_items:
                table_header = "| Item | Qty | Price | Subtotal |\n| :--- | :--- | :--- | :--- |\n"
                cart_table.markdown(table_header + "\n".join(bill_items))
                total_display.markdown(f"## Total Amount: ₹{grand_total:.2f}")
        else:
            cart_table.info("Bring an item into the camera view...")
            total_display.write("")

# 6. Final Checkout
if st.button("Finalize & Save Transaction", type="primary", use_container_width=True):
    if 'grand_total' in locals() and grand_total > 0:
        # You can add logic here to INSERT into a 'transactions' table
        st.balloons()
        st.success(f"Successfully billed ₹{grand_total:.2f}")
    else:
        st.warning("No items detected to bill.")
