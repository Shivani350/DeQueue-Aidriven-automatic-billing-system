import streamlit as st
import cv2
import sqlite3
from ultralytics import YOLO
import datetime
import numpy as np
from PIL import Image

# 1. Page & Session Setup
st.set_page_config(page_title="AI Smart Checkout", layout="wide")

# 2. Database Functions
def get_db_info(item_name):
    try:
        conn = sqlite3.connect('mall.db')
        cursor = conn.cursor()
        cursor.execute("SELECT barcode, price FROM products WHERE name = ?", (item_name,))
        result = cursor.fetchone()
        conn.close()
        return result
    except:
        return None

# 3. UI Layout
st.title("🛒 AI Smart Checkout MVP")

# Load AI Model (Cached to prevent timeouts)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Use Image Uploader for Cloud MVP
uploaded_file = st.file_uploader("📸 Upload an item photo to scan", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Run AI Detection
    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()
    detected_items = [model.names[int(c)] for c in results[0].boxes.cls]

    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detected Items")
        st.image(annotated_frame, channels="BGR")

    with col2:
        st.subheader("Shopping Cart")
        current_bill = ""
        total_val = 0.0
        
        for item in set(detected_items):
            db_data = get_db_info(item)
            if db_data:
                barcode, price = db_data
                qty = detected_items.count(item)
                current_bill += f"**{item.capitalize()}** x{qty} — ₹{price * qty:.2f} \n"
                total_val += (price * qty)

        if current_bill:
            st.markdown(current_bill)
            st.divider()
            st.markdown(f"### Total: ₹{total_val:.2f}")
            if st.button("Finalize Sale", type="primary"):
                st.balloons()
                st.success("Transaction Complete!")
        else:
            st.warning("No recognized items found in the database.")
