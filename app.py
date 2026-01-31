import streamlit as st
import cv2
import sqlite3
from ultralytics import YOLO
import datetime

# 1. Page & Session Setup
st.set_page_config(page_title="AI Smart Checkout", layout="wide")

if 'checkout_complete' not in st.session_state:
    st.session_state.checkout_complete = False

# 2. Database Functions
def get_db_info(item_name):
    conn = sqlite3.connect('mall.db')
    cursor = conn.cursor()
    cursor.execute("SELECT barcode, price FROM products WHERE name = ?", (item_name,))
    result = cursor.fetchone()
    conn.close()
    return result

def commit_transaction(total, count):
    conn = sqlite3.connect('mall.db')
    cursor = conn.cursor()
    # Ensure this table exists in your mall.db
    cursor.execute("INSERT INTO transactions (total_amount, items_count) VALUES (?, ?)", (total, count))
    conn.commit()
    conn.close()

# 3. Dialog (The Pop-up Bill)
@st.dialog("🧾 FINAL RECEIPT")
def show_receipt(bill_text, total, count):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown("### **VISTA SMART STORE**")
    st.text(f"Date: {now}")
    st.divider()
    st.markdown(bill_text)
    st.divider()
    st.markdown(f"**Total Items:** {count}")
    st.markdown(f"## **Total Amount: ₹{total:.2f}**")
    st.divider()
    if st.button("Start Next Sale", use_container_width=True):
        st.session_state.checkout_complete = False
        st.rerun()

# 4. UI Layout
st.title("🛒 AI Smart Checkout Terminal")
col_video, col_billing = st.columns([2, 1])

# Load AI
model = YOLO('yolov8n.pt')
video_feed = cv2.VideoCapture(0)

with col_video:
    st.subheader("Live Scanning Feed")
    frame_window = st.empty()

with col_billing:
    st.subheader("Shopping Cart")
    cart_display = st.empty()
    checkout_btn = st.button("Finalize Sale", type="primary", use_container_width=True)

# 5. The Main Loop
while video_feed.isOpened():
    ret, frame = video_feed.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()
    detected_items = [model.names[int(c)] for c in results[0].boxes.cls]

    frame_window.image(annotated_frame, channels="BGR")

    current_bill = ""
    total_val = 0.0
    item_count = 0

    for item in set(detected_items):
        db_data = get_db_info(item)
        if db_data:
            barcode, price = db_data
            qty = detected_items.count(item)
            current_bill += f"**{item.capitalize()}** x{qty} — ₹{price * qty:.2f}  \n"
            total_val += (price * qty)
            item_count += qty

    cart_display.markdown(f"{current_bill} \n---\n ### Total: ₹{total_val:.2f}")

    # --- CRITICAL CHANGE: Logic moved INSIDE the loop ---
    if checkout_btn:
        if item_count > 0:
            commit_transaction(total_val, item_count)
            video_feed.release()
            show_receipt(current_bill, total_val, item_count)
            st.balloons()
            break
        else:
            st.sidebar.warning("Scan an item first!")