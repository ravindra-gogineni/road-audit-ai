import streamlit as st
import cv2
import time
import os
import math
import geocoder 
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from ultralytics import YOLO
from fpdf import FPDF
import base64
import requests
import tempfile

# ==========================================
# ⚙️ CONFIGURATION & CSS
# ==========================================
MODEL_PATH = "best.pt"
DEFAULT_VIDEO = "demo.mp4"

# BREVO API (Securely set via Streamlit Secrets for Public Deployment)
BREVO_API_KEY = st.secrets["BREVO_API_KEY"] if "BREVO_API_KEY" in st.secrets else "PASTE_LOCAL_KEY_HERE"
SENDER_NAME = "Citizen Road Reporter"
SENDER_EMAIL = "revathinalluri999@gmail.com"

st.set_page_config(page_title="Citizen Road Reporter", layout="wide", page_icon="🛣️", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2f;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #33334d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        text-align: center;
        margin-bottom: 24px;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-title { font-size: 1rem; color: #a1a1aa; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-count { font-size: 3rem; color: #10b981; font-weight: 800; line-height: 1; }
    .metric-cost { font-size: 3rem; color: #ef4444; font-weight: 800; line-height: 1; }
    
    /* Responsive Font Adjustments */
    @media (max-width: 768px) {
        .metric-count, .metric-cost { font-size: 2.2rem; }
        .metric-card { padding: 16px; margin-bottom: 16px; }
        .stMarkdown div p { font-size: 15px; }
    }
    
    /* Touch Friendly Buttons */
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 10px;
    }
    
    div.block-container { padding-top: 2rem; padding-bottom: 2rem; }
    header { visibility: visible; } 
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 CORE LOGIC & EMAIL
# ==========================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model missing: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

class RoadAuditState:
    def __init__(self):
        self.processed_ids = set()
        self.processed_centroids = []
        self.detections = []
        self.total_cost = 0
        self.pothole_count = 0
        self.severity_counts = {"MINOR": 0, "MODERATE": 0, "CRITICAL": 0}

    def calculate_severity(self, box, frame_area):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        percent_area = ((width * height) / frame_area) * 100
        
        # --- FINAL "PERFECT DEMO" CALIBRATION (REDUCED COSTS) ---
        # Adjusted to keep total repair bills around 15k-25k for the demo video.
        base_cost = 200
        if percent_area < 1.0: 
            return "MINOR", int(base_cost + percent_area * 100), (16, 185, 129) # Green
        elif 1.0 <= percent_area < 5.0: 
            return "MODERATE", int(base_cost + percent_area * 300), (245, 158, 11) # Orange
        else: 
            return "CRITICAL", int(base_cost + percent_area * 1000), (239, 68, 68) # Red

    def is_duplicate(self, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for (old_cx, old_cy) in self.processed_centroids:
            if math.sqrt((cx-old_cx)**2 + (cy-old_cy)**2) < 50:
                return True, cx, cy
        return False, cx, cy

def create_pdf(detections, total_cost, count, reporter_name, road_name):
    if not detections:
        return None
    report_name = f"Audit_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CITIZEN ROAD DAMAGE REPORT", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Date Submitted: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Submitted By: {reporter_name if reporter_name else 'Anonymous Citizen'}", ln=True)
    pdf.cell(0, 8, f"Target Area: {road_name if road_name else 'Unknown Location'}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Total Potholes Detected: {count}", ln=True)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, f"ESTIMATED REPAIR COST: Rs. {total_cost:,}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=10)
    for det in detections:
        pdf.cell(0, 8, f"Defect ID #{det['ID']} | Severity: {det['Severity']} | Cost: Rs.{det['Cost (Rs)']}", ln=True)
        pdf.cell(0, 8, f"   GPS Coordinates: {det['Location']}", ln=True)
        pdf.ln(3)
        
    pdf.output(report_name)
    return report_name

def send_brevo_email(report_name, total_cost, target_email, reporter_name):
    try:
        with open(report_name, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        url = "https://api.brevo.com/v3/smtp/email"
        headers = {"accept": "application/json", "api-key": BREVO_API_KEY, "content-type": "application/json"}
        payload = {
            "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
            "to": [{"email": target_email}],
            "subject": f"Citizen Report: Urgent Road Repairs Needed (Est Rs.{total_cost:,})",
            "htmlContent": f"<h2>Citizen Road Damage Report</h2><p>This report was automatically submitted by {reporter_name}. Please find attached the AI-generated repair estimate and location details.</p>",
            "attachment": [{"content": pdf_base64, "name": report_name}]
        }
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 201: return True, "Invoice successfully sent to authorities!"
        return False, f"Email failed: {res.text}"
    except Exception as e:
        return False, str(e)


# ==========================================
# 🎛️ SIDEBAR / SAAS CONTROLS
# ==========================================

st.sidebar.title("📸 Input Source")
source_type = st.sidebar.radio("Location", ["Demo Video", "Upload Video", "Live Camera"], index=0)

video_path = None
if source_type == "Demo Video":
    video_path = DEFAULT_VIDEO
elif source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload an MP4, AVI, or MOV file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif source_type == "Live Camera":
    video_path = 0

st.sidebar.divider()
start_btn = st.sidebar.button("🚀 Start Scan", use_container_width=True, type="primary")
stop_btn = st.sidebar.button("🛑 Stop Scan", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("📝 Dispatch Details")
input_name = st.sidebar.text_input("Your Name / Organization", placeholder="e.g. John Doe")
input_road = st.sidebar.text_input("Road Name / Region", placeholder="e.g. Main Street, Sector 4")
input_email = st.sidebar.text_input("Authority Email", placeholder="pwd@gov.in")
send_btn = st.sidebar.button("📧 Send Report to Authority", use_container_width=True)

# Initialize Session State
if "audit_state" not in st.session_state:
    st.session_state.audit_state = RoadAuditState()
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_chart_count" not in st.session_state:
    st.session_state.last_chart_count = -1

# Button Logic
if start_btn: 
    # Reset everything for a fresh scan
    st.session_state.audit_state = RoadAuditState()
    st.session_state.last_chart_count = -1
    st.session_state.is_running = True
    
if stop_btn: st.session_state.is_running = False

audit = st.session_state.audit_state

# Handle Manual Email Trigger
if send_btn:
    if audit.pothole_count > 0 and input_email:
        with st.sidebar.status("Generating Report..."):
            pdf_path = create_pdf(audit.detections, audit.total_cost, audit.pothole_count, input_name, input_road)
            success, msg = send_brevo_email(pdf_path, audit.total_cost, input_email, input_name)
            os.remove(pdf_path) # cleanup
        if success: st.sidebar.success(msg)
        else: st.sidebar.error(msg)
    elif not input_email:
        st.sidebar.warning("Please enter the Authority Email first.")
    else:
        st.sidebar.warning("No potholes scanned yet. Nothing to send.")

# ==========================================
# 🖥️ DASHBOARD UI
# ==========================================

st.title("🛣️ Citizen Road Reporter Dashboard")
st.caption("AI-Powered Road Audit & Automated Reporting System")

# Responsive Main Layout using Tabs
tab_live, tab_analytics = st.tabs(["🔴 Live Inspection", "📊 Analytics & Reports"])

# Data Storage Elements (Placed once to be accessed inside tabs)
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
record_table = st.empty()

with tab_live:
    live_container = st.container()
    with live_container:
        st.subheader("Inspection Feed")
        video_placeholder = st.empty()
        if not st.session_state.is_running:
            st.info("👈 Use the sidebar to select your video source and click **Start Scan** to begin.")

with tab_analytics:
    analytics_container = st.container()
    with analytics_container:
        st.subheader("Aggregated Data & Reporting")
        
        # Grid for Metrics and Export Buttons
        m_col1, m_col2 = st.columns([1, 1])
        with m_col1:
            st.markdown("### Export Tools")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if audit.detections:
                    df = pd.DataFrame(audit.detections)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", data=csv, file_name="potholes.csv", mime="text/csv")
                else:
                    st.download_button("📥 Download CSV", data="", disabled=True)
            with export_col2:
                if audit.detections:
                    pdf_file = create_pdf(audit.detections, audit.total_cost, audit.pothole_count, input_name, input_road)
                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Export PDF", f, file_name="road_report.pdf")
                    os.remove(pdf_file)
                else:
                    st.download_button("📄 Export PDF", data="", disabled=True)
            
        with m_col2:
            # Metrics will be injected here via the placeholder
            pass
            
        st.divider()
        
        # Display current audit table
        st.markdown("### Detailed Pothole Log")
        # record_table is already an empty placeholder, but we will put it in an expander for mobile
        with st.expander("View Full Detection Log", expanded=True):
            # The placeholder 'record_table' will be populated by the loop
            pass

def update_ui_elements(audit):
    # Metrics
    metrics_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DEFECTS LOGGED</div>
            <div class="metric-count">{audit.pothole_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">ESTIMATED REPAIR BILL</div>
            <div class="metric-cost">₹ {audit.total_cost:,}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Pie Chart - Only update when count changes to avoid duplicate ID errors and lag
    if audit.pothole_count > 0 and audit.pothole_count != st.session_state.last_chart_count:
        st.session_state.last_chart_count = audit.pothole_count
        labels = list(audit.severity_counts.keys())
        values = list(audit.severity_counts.values())
        fig = px.pie(names=labels, values=values, title="Severity Distribution", hole=0.4, 
                     color=labels, color_discrete_map={"MINOR":"#10b981", "MODERATE":"#f59e0b", "CRITICAL":"#ef4444"})
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a1a1aa'))
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"severity_chart_{audit.pothole_count}")
        
    # Table
    if audit.detections:
        df = pd.DataFrame(audit.detections)
        record_table.dataframe(df.tail(10), use_container_width=True)

# Initial Render
update_ui_elements(audit)

# Video Execution Logic
if not st.session_state.is_running:
    # Notice: the info message is already shown inside tab_live in the layout section
    pass
else:
    model = load_model()
    # Verify file existence if it's not a webcam
    if source_type != "Live Camera" and not os.path.exists(video_path):
        st.error(f"⚠️ Video File Not Found: {video_path}. Please ensure you have uploaded it!")
        st.session_state.is_running = False
    else:
        # Special handling for Camera Access
        if source_type == "Live Camera":
            # Try to open webcam (0 is usually the default)
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            if source_type == "Live Camera":
                st.error("📷 **Camera Access Failed**: This usually happens on Streamlit Cloud (which doesn't support direct webcam) or if your camera is being used by another app. **Please use 'Upload Video' for the smoothest demo experience!**")
            else:
                st.error(f"⚠️ **Error opening file**: {video_path}. Please check if the file format is supported (.mp4, .avi).")
            st.session_state.is_running = False
        else:
            # --- SMART GPS LOGIC ---
            # Try to get live location, otherwise use the Road Name from the sidebar
            try:
                latlng = geocoder.ip('me').latlng
                session_gps = f"{latlng[0]:.5f}, {latlng[1]:.5f}" if latlng else input_road if input_road else "📍 Hyderabad, IN"
            except:
                session_gps = input_road if input_road else "📍 Hyderabad, IN"
            
            frame_count = 0 
            while cap.isOpened() and st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    st.success("Video processing complete! You can now send the report.")
                    st.session_state.is_running = False
                    break
            
                frame_count += 1
                
                # Convert to RGB once for AI and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_area = frame.shape[0] * frame.shape[1]
                
                # --- ACCURACY PRIORITY ---
                # Run the AI on EVERY FRAME so we never miss a pothole!
                results = model.track(frame_rgb, persist=True, tracker="botsort.yaml", conf=0.4, verbose=False)
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, ids):
                        severity, cost, color = audit.calculate_severity(box, frame_area)
                        
                        if track_id not in audit.processed_ids:
                            is_dup, cx, cy = audit.is_duplicate(box)
                            if not is_dup:
                                audit.processed_ids.add(track_id)
                                audit.processed_centroids.append((cx, cy))
                                audit.pothole_count += 1
                                audit.total_cost += cost
                                audit.severity_counts[severity] += 1
                                
                                audit.detections.append({
                                    "ID": track_id,
                                    "Severity": severity,
                                    "Cost (Rs)": cost,
                                    "Location": session_gps
                                })
                        
                        # Draw boxes on the RGB frame for display
                        cv2.rectangle(frame_rgb, (box[0], box[1]), (box[2], box[3]), color, 3)
                        cv2.putText(frame_rgb, f"ID:{track_id} {severity}", (box[0], box[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- SMOOTHNESS PRIORITY ---
                # We only send 1 frame to the web for every 5 AI scans.
                # This makes the video stream light and fast!
                if frame_count % 5 == 0:
                    try:
                        # Ultra-low res for cloud speed
                        display_frame = cv2.resize(frame_rgb, (480, 270))
                        video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                        update_ui_elements(audit) # Update the stats/table live with the video
                    except:
                        pass
                    
                time.sleep(0.01)
                    
            cap.release()
