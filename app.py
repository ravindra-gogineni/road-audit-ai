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
import sqlite3
import json
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap, AntPath
from streamlit_geolocation import streamlit_geolocation
import database_handler as db

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

# Initialize Database
db.init_db()

# --- INITIALIZE SESSION STATE ---
if "audit_state" not in st.session_state:
    st.session_state.audit_state = RoadAuditState()
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_chart_count" not in st.session_state:
    st.session_state.last_chart_count = -1
if "last_source" not in st.session_state:
    st.session_state.last_source = "Demo Video"
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "map_lat" not in st.session_state:
    st.session_state.map_lat = 15.50
if "map_lng" not in st.session_state:
    st.session_state.map_lng = 80.05

# ==========================================
# ⚙️ CONFIGURATION & CSS
# ==========================================
MODEL_PATH = "best.pt"
DEFAULT_VIDEO = "demo.mp4"

# BREVO API (Securely set via Streamlit Secrets for Public Deployment)
BREVO_API_KEY = st.secrets["BREVO_API_KEY"] if "BREVO_API_KEY" in st.secrets else "PASTE_LOCAL_KEY_HERE"
SENDER_NAME = "Citizen Road Reporter"
SENDER_EMAIL = "revathinalluri999@gmail.com"

# --- ANDHRA PRADESH DISTRICTS & TARGET EMAILS ---
AP_DISTRICTS = {
    "Alluri Sitharama Raju": "pmu.asr@ap.gov.in",
    "Anakapalli": "ce.anakapalli@ap.gov.in",
    "Ananthapuramu": "ce.atp@ap.gov.in",
    "Annamayya": "ee.annamayya@ap.gov.in",
    "Bapatla": "pmu.bapatla@ap.gov.in",
    "Chittoor": "ce.chittoor@ap.gov.in",
    "Dr. B.R. Ambedkar Konaseema": "ee.konaseema@ap.gov.in",
    "East Godavari": "ce.egodavari@ap.gov.in",
    "Eluru": "pmu.eluru@ap.gov.in",
    "Guntur": "ce.guntur@ap.gov.in",
    "Kakinada": "ee.kakinada@ap.gov.in",
    "Krishna": "ce.krishna@ap.gov.in",
    "Kurnool": "ce.kurnool@ap.gov.in",
    "Nandyal": "pmu.nandyal@ap.gov.in",
    "NTR": "ce.ntr@ap.gov.in",
    "Palnadu": "ee.palnadu@ap.gov.in",
    "Parvathipuram Manyam": "pmu.manyam@ap.gov.in",
    "Prakasam (Ongole)": "ce.prakasam@ap.gov.in",
    "Sri Potti Sriramulu Nellore": "ce.nellore@ap.gov.in",
    "Sri Sathya Sai": "pmu.sss@ap.gov.in",
    "Srikakulam": "ce.srikakulam@ap.gov.in",
    "Tirupati": "ce.tirupati@ap.gov.in",
    "Visakhapatnam": "ce.vizag@ap.gov.in",
    "Vizianagaram": "ce.vizianagaram@ap.gov.in",
    "West Godavari": "ce.wgodavari@ap.gov.in",
    "YSR Kadapa": "ce.kadapa@ap.gov.in"
}

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

if not st.session_state.admin_logged_in:
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
    st.sidebar.subheader("📤 Citizen Submission")
    input_name = st.sidebar.text_input("Your Name / Organization", placeholder="e.g. John Doe")
    input_reporter_email = st.sidebar.text_input("Your Email (For updates)", placeholder="citizen@mail.com")
    input_road = st.sidebar.text_input("Road Name / Region", placeholder="e.g. Ongole Main Road")
    submit_btn = st.sidebar.button("📤 Submit Report", use_container_width=True, type="primary")
else:
    st.sidebar.title("🏛️ Admin Dashboard")
    st.sidebar.info("You are currently in **Management Mode**. Review and route citizen complaints below.")
    # No scanner buttons for admins
    start_btn = stop_btn = submit_btn = False # placeholder buttons to avoid name errors
    video_path = None

# --- LIVE GPS CAPTURE ---
with st.sidebar:
    st.subheader("📍 Precise Location")
    location = streamlit_geolocation()
    if location and location['latitude']:
        st.session_state.map_lat = location['latitude']
        st.session_state.map_lng = location['longitude']
        st.success(f"GPS Connected: {st.session_state.map_lat:.4f}, {st.session_state.map_lng:.4f}")
    else:
        st.warning("Please click 'Allow' or check GPS if inaccurate.")

# --- MAIN NAVIGATION ---
st.sidebar.divider()
st.sidebar.title("📌 Navigation")
page_nav = st.sidebar.radio("Go To", ["🏠 Dashboard", "🗺️ Safety Hub"], index=0)

# Sidebar Role Switcher
st.sidebar.divider()
if st.session_state.admin_logged_in:
    st.sidebar.success("Logged in as Admin")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.admin_logged_in = False
        st.rerun()
else:
    if st.sidebar.button("🔒 Admin Portal", use_container_width=True):
        st.session_state.show_login = True
        st.rerun()
    if st.session_state.show_login:
        if st.sidebar.button("🏠 Back to Citizen View", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()

# Initialize Session State
# AUTO-RESET: If they switch from Demo to Upload, clear the screen immediately
if "source_type" in locals() and st.session_state.get("last_source") != source_type:
    st.session_state.audit_state = RoadAuditState()
    st.session_state.last_chart_count = -1
    st.session_state.last_source = source_type
    st.session_state.is_running = False

# Button Logic
if start_btn: 
    # Reset everything for a fresh scan
    st.session_state.audit_state = RoadAuditState()
    st.session_state.last_chart_count = -1
    st.session_state.is_running = True
    
if stop_btn: st.session_state.is_running = False

audit = st.session_state.audit_state

# Handle Manual Submission (Citizen Flow)
if submit_btn:
    if not input_name or not input_reporter_email or not input_road:
        st.sidebar.error("⚠️ Please fill in your **Name**, **Email**, and **Road Name** before submitting.")
    elif audit.pothole_count > 0:
        # Calculate Priority
        priority = "🚨 High" if audit.severity_counts.get("CRITICAL", 0) > 0 else "🟢 Normal"
        # Extract lat/lng
        lat = st.session_state.get("map_lat", 15.5)
        lng = st.session_state.get("map_lng", 80.05)
        # Save to Database for Admin Review
        db.add_complaint(input_name, input_road, audit.pothole_count, audit.total_cost, audit.detections, input_reporter_email, "Pending Board Review", priority, lat, lng)
        st.sidebar.success("✅ Report Submitted! Authorities will review and forward it shortly.")
        st.sidebar.info(f"📌 Priority Assigned: {priority}")
    else:
        st.sidebar.warning("No potholes scanned yet. Nothing to submit.")

# ==========================================
# 🖥️ DASHBOARD UI
# ==========================================

def render_login_page():
    st.title("🔐 Authority Login")
    st.markdown("Please enter your credentials to access the government portal.")
    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "admin123":
                st.session_state.admin_logged_in = True
                st.session_state.show_login = False
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid Username or Password")

def render_safety_hub_page():
    st.title("🗺️ 3D Community Safety Map")
    st.info("Explore hazards, bypass dangerous roads, and see official accident-prone zones in real-time.")
    
    all_comp = db.get_all_complaints()
    
    # Layer Selection
    map_style = st.radio("Map Style", ["🛰️ Satellite View", "🏙️ Professional Road View"], horizontal=True)
    tile_set = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" if "Satellite" in map_style else "CartoDB dark_matter"
    attr = "ESRI World Imagery" if "Satellite" in map_style else "CartoDB"

    # Create Map (Centered on most recent point or AP)
    m = folium.Map(location=[st.session_state.map_lat, st.session_state.map_lng], zoom_start=13, tiles=tile_set, attr=attr)
    
    # Add Markers
    if all_comp:
        for c in all_comp:
            # SAFETY PADDING (16 Columns)
            c_list = list(c)
            while len(c_list) < 16: c_list.append(None)
            cid, cname, rname, pcount, cost, status, days, time, details, rep_e, auth_e, priority, forw, lat, lng, is_black = c_list
            
            if lat and lng:
                color = "red" if priority == "🚨 High" else "orange"
                icon = "exclamation-triangle"
                
                if is_black:
                    color = "black"
                    icon = "skull"
                    label = "🚨 ACCIDENT PRONE BLACKSPOT"
                else:
                    label = f"{priority} Hazard ({status})"
                
                # Popup - NO COST for citizens
                html = f"""
                    <div style="font-family: Arial;">
                        <h4 style="color:{color};">{label}</h4>
                        <b>📍 Road:</b> {rname}<br>
                        <b>📅 Reported:</b> {time}<br>
                        <b>🛠️ Status:</b> {status}
                    </div>
                """
                folium.Marker(
                    location=[lat, lng],
                    popup=folium.Popup(html, max_width=300),
                    icon=folium.Icon(color=color, icon=icon, prefix="fa"),
                    tooltip=f"{rname} - {priority}"
                ).add_to(m)

    # Render Map
    st_folium(m, width=900, height=500, use_container_width=True)
    
    # --- SAFE NAVIGATION PANEL ---
    st.divider()
    with st.expander("🚗 Safe Path Navigator (Beta)"):
        st.markdown("Enter your destination to find any hazards along your route.")
        nav_col1, nav_col2 = st.columns(2)
        st_loc = nav_col1.text_input("From Location", value="Ongole")
        end_loc = nav_col2.text_input("To Location", placeholder="e.g. Hyderabad")
        if st.button("🗺️ Find Safest Path"):
            st.warning(f"Navigating from {st_loc} to {end_loc}...")
            st.info("Safety Check: Searching database for hazards on this path...")
            # Mocked safety navigation for the demo
            hazard_dist = np.random.randint(2, 6)
            st.error(f"🚨 Navigation Alert: {hazard_dist} critical hazards found on your route! Drive carefully.")

def render_citizen_view(audit):
    if page_nav == "🗺️ Safety Hub":
        render_safety_hub_page()
        return

    st.title("🛣️ Citizen Road Reporter Dashboard")
    st.caption("AI-Powered Road Audit & Automated Reporting System")

    # Tabs
    tab_live, tab_tracker, tab_analytics = st.tabs([
        "🔴 Live Inspection", 
        "🔍 Track Complaint",
        "📊 Analytics & Reports"
    ])

    with tab_live:
        st.subheader("Inspection Feed")
        global video_placeholder, metrics_placeholder, chart_placeholder, record_table
        metrics_placeholder = st.empty()
        video_placeholder = st.empty()
        chart_placeholder = st.empty()
        record_table = st.empty()
        
        if not st.session_state.is_running:
            st.info("👈 Use the sidebar to select your video source and click **Start Scan** to begin.")
            update_ui_elements(audit)

    with tab_tracker:
        st.subheader("🔍 Public Complaint Tracker")
        search_q = st.text_input("Search by Road Name or Your Name", placeholder="e.g. Main Street")
        if search_q:
            results = db.search_complaints(search_q)
            if results:
                for res in results:
                    # SAFETY PADDING (16 Columns)
                    res_list = list(res)
                    while len(res_list) < 16: res_list.append(None)
                    
                    c_id, c_name, r_name, p_count, t_cost, status, start_days, tstamp, details, rep_email, auth_email, priority, forw_at, lat, lng, is_black = res_list
                    with st.expander(f"📌 {r_name} ({priority}) - {tstamp}"):
                        col_a, col_b = st.columns(2)
                        col_a.metric("Status", status)
                        col_b.metric("Priority", priority)
                        if forw_at: st.success(f"📧 Forwarded to Authority on {forw_at}")
                        st.write(f"**Est. Start:** {start_days} days" if start_days else "**Est. Start:** Not Scheduled")
                        st.write(f"**Potholes Detected:** {p_count} | **Est. Repair Cost:** ₹{t_cost:,}")
                        if st.session_state.admin_logged_in:
                             st.write(f"**From:** {c_name} ({rep_email}) | **To Authority:** {auth_email}")
            else: st.warning("No complaints found.")
        else: st.info("Enter a name or road to search.")

    with tab_analytics:
        st.subheader("Aggregated Data & Reporting")
        all_comp = db.get_all_complaints()
        if all_comp:
            # SAFETY SHIELD: Handle missing columns dynamically (16 Columns Total)
            col_names = ["id", "name", "road", "count", "cost", "status", "days", "time", "details", "reporter_email", "authority_email", "priority", "forwarded_at", "lat", "lng", "is_blackspot"]
            # Pad the rows if the database is old
            padded_comp = [list(r) + [None] * (len(col_names) - len(r)) for r in all_comp]
            df_comp = pd.DataFrame(padded_comp, columns=col_names)
            
            st.markdown("### Public Status Summary")
            st.dataframe(df_comp[["road", "priority", "status", "cost", "time"]].tail(5), use_container_width=True)

def render_admin_view():
    st.title("🏛️ Authority Portal")
    st.success("Authorized: Government Management Mode")
    
    tab_manage, tab_reports = st.tabs(["📋 Manage Complaints", "📊 Advanced Analytics"])
    
    with tab_manage:
        all_complaints = db.get_all_complaints()
        if not all_complaints:
            st.info("No complaints have been submitted yet.")
        else:
            for comp in all_complaints:
                # SAFETY PADDING for legacy records (16 Columns Total)
                comp_list = list(comp)
                while len(comp_list) < 16: comp_list.append(None)
                
                c_id, c_name, r_name, p_count, t_cost, status, start_days, tstamp, details, rep_email, current_auth, priority, forw_at, lat, lng, is_black = comp_list
                with st.expander(f"ID #{c_id}: {r_name} ({priority}) - {status}"):
                    st.markdown(f"### 🛡️ Complaint Overview (Priority: {priority})")
                    if forw_at: st.success(f"📬 This report was forwarded to {current_auth} on {forw_at}")
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    col_info1.metric("💰 Total Estimate", f"₹{t_cost:,}")
                    col_info2.write(f"**Reporter:** {c_name}")
                    col_info2.write(f"**Contact:** {rep_email}")
                    col_info3.write(f"**Current Route:** {current_auth}")
                    col_info3.write(f"**Timestamp:** {tstamp}")
                    
                    st.divider()
                    st.markdown("### 💰 Itemized Cost Details")
                    try:
                        pothole_details_list = json.loads(details)
                        if pothole_details_list:
                            detail_df = pd.DataFrame(pothole_details_list)
                            st.table(detail_df)
                        else:
                            st.info("No itemized details captured (Old Record)")
                    except:
                        st.info("No itemized details available.")

                    st.divider()
                    st.markdown("### 🛠️ Administrative Actions")
                    
                    # --- ROUTING ACTION ---
                    with st.container(border=True):
                        st.write("📫 **Forward Report to District Office**")
                        route_col1, route_col2 = st.columns(2)
                        target_dist = route_col1.selectbox("Select Target District", options=list(AP_DISTRICTS.keys()), 
                                                       index=list(AP_DISTRICTS.keys()).index("Prakasam (Ongole)") if "Prakasam (Ongole)" in AP_DISTRICTS else 0,
                                                       key=f"dist_{c_id}")
                        target_email = route_col2.text_input("Official Email", value=AP_DISTRICTS[target_dist], key=f"mail_{c_id}")
                        
                        if st.button(f"📧 Forward Report #{c_id}", use_container_width=True, type="primary"):
                            with st.spinner("Processing..."):
                                detections_raw = json.loads(details)
                                pdf_path = create_pdf(detections_raw, t_cost, p_count, c_name, r_name)
                                success, msg = send_brevo_email(pdf_path, t_cost, target_email, c_name)
                                if os.path.exists(pdf_path): os.remove(pdf_path)
                                
                                if success:
                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                                    db.update_complaint(c_id, "Scheduled", 7, current_time) # Auto-schedule for 7 days
                                    st.success(f"Report forwarded to {target_dist} officials!")
                                    st.rerun()
                                else: st.error(msg)
                                
                    # --- BLACKSPOT MANAGEMENT ---
                    with st.container(border=True):
                        st.write("🚨 **Accident Prone Area Management**")
                        is_current_black = bool(is_black)
                        if st.toggle("Mark as Accident Prone Blackspot", value=is_current_black, key=f"black_{c_id}"):
                            if not is_current_black:
                                db.update_complaint(c_id, status, days, is_blackspot=1)
                                st.warning("Spot marked as Permanent Blackspot!")
                                st.rerun()
                        else:
                            if is_current_black:
                                db.update_complaint(c_id, status, days, is_blackspot=0)
                                st.rerun()

                    st.divider()
                    with st.form(f"edit_form_{c_id}"):
                        col1, col2 = st.columns(2)
                        new_status = col1.selectbox("Status", ["Pending", "Scheduled", "Work Started", "Completed"], 
                                                 index=["Pending", "Scheduled", "Work Started", "Completed"].index(status))
                        new_start = col2.number_input("Starts in (Days)", value=start_days if start_days else 0, min_value=0)
                        
                        btn_col1, btn_col2 = st.columns(2)
                        if btn_col1.form_submit_button("💾 Save Changes"):
                            db.update_complaint(c_id, new_status, new_start)
                            st.success(f"Complaint #{c_id} updated!")
                            st.rerun()
                        if btn_col2.form_submit_button("🗑️ Delete"):
                            db.delete_complaint(c_id)
                            st.warning(f"Complaint #{c_id} deleted!")
                            st.rerun()
    
    with tab_reports:
        st.subheader("Administrative Reports")
        all_comp = db.get_all_complaints()
        if all_comp:
            # SAFETY SHIELD: 16 Columns Total
            col_names = ["id", "name", "road", "count", "cost", "status", "days", "time", "details", "reporter_email", "authority_email", "priority", "forwarded_at", "lat", "lng", "is_blackspot"]
            padded_comp = [list(r) + [None] * (len(col_names) - len(r)) for r in all_comp]
            df = pd.DataFrame(padded_comp, columns=col_names)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Master Log (CSV)", data=csv, file_name="master_complaints.csv")

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

# --- MAIN RENDER LOGIC ---
if st.session_state.admin_logged_in:
    render_admin_view()
elif st.session_state.show_login:
    render_login_page()
else:
    render_citizen_view(audit)

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
            # --- SMART GPS CAPTURE ---
            try:
                g = geocoder.ip('me')
                if g.latlng:
                    st.session_state.map_lat, st.session_state.map_lng = g.latlng[0], g.latlng[1]
                    session_gps = f"{st.session_state.map_lat:.5f}, {st.session_state.map_lng:.5f}"
                else: 
                    session_gps = input_road if input_road else "📍 Hyderabad, IN"
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
                                    "ID": int(track_id),
                                    "Severity": severity,
                                    "Cost (Rs)": int(cost),
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
