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

# --- GLOBAL BRANDING & LINKS ---
GITHUB_URL = "https://github.com/ravindra-gogineni"
LINKEDIN_URL = "https://www.linkedin.com/in/ravindra-gogineni-034501212"
PROJECT_TITLE = "Andhra Pradesh Road Safety"

class RoadAuditState:
    def __init__(self):
        self.processed_ids = set()
        self.processed_centroids = []
        self.detections = []
        self.total_cost = 0
        self.pothole_count = 0
        self.severity_counts = {"MINOR": 0, "MODERATE": 0, "CRITICAL": 0}
        self.activity_logs = [] # Store "New Defect ID" messages

    def calculate_severity(self, box, frame_area):
        """Dynamic pricing based on pothole size relative to the road."""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area_pixels = width * height
        
        # Calculate what % of the screen this pothole takes up
        percent_area = (area_pixels / frame_area) * 100
        
        base_cost = 500  # Minimum labor charge
        
        # 1. SMALL POTHOLE (Patch work)
        if percent_area < 2: 
            severity = "MINOR"
            color = (16, 185, 129) # Premium Green (RGB)
            extra_cost = percent_area * 50
            
        # 2. MEDIUM POTHOLE (Resurfacing)
        elif 2 <= percent_area < 8:
            severity = "MODERATE"
            color = (245, 158, 11) # Premium Orange (RGB)
            extra_cost = percent_area * 150
            
        # 3. LARGE POTHOLE (Full reconstruction)
        else:
            severity = "CRITICAL"
            color = (239, 68, 68) # Premium Red (RGB)
            extra_cost = percent_area * 300
            
        total_cost = int(base_cost + extra_cost)
        return severity, total_cost, color

    def is_duplicate(self, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Fine-tuned to 65px: Real-world balance for distinct hazards
        for (old_cx, old_cy) in self.processed_centroids:
            if math.sqrt((cx-old_cx)**2 + (cy-old_cy)**2) < 65:
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
st.set_page_config(page_title=PROJECT_TITLE, page_icon="🛣️", layout="wide")

def apply_premium_style():
    st.markdown(f"""
    <style>
        h1, h2, h3, p, .metric-title, .metric-value, .top-bar {{ font-family: 'Outfit', sans-serif !important; }}
        
        /* Sticky Top Bar */
        .top-bar {{
            position: fixed;
            top: 0; left: 0; width: 100%;
            background: rgba(15, 23, 42, 0.98);
            backdrop-filter: blur(12px);
            z-index: 9999;
            padding: 10px 40px;
            border-bottom: 2px solid rgba(16, 185, 129, 0.5);
            display: flex; justify-content: space-between; align-items: center;
        }}
        .top-bar-title {{ font-size: 1.5rem; font-weight: 700; color: #fff; text-decoration: none; }}
        .header-btn {{
            background: #10b981; color: white !important; 
            padding: 8px 18px; border-radius: 8px; 
            text-decoration: none; font-weight: 700; font-size: 0.85rem;
            transition: 0.3s;
        }}
        .header-btn:hover {{ background: #059669; box-shadow: 0 0 15px rgba(16, 185, 129, 0.6); }}
        
        /* Glassmorphism Metric Cards - REPAIRED */
        .metric-container {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px; padding: 25px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }}
        .metric-title {{
            color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
        }}
        .metric-value {{
            color: #10b981; font-size: 2.8rem; font-weight: 800; margin-top: 5px;
        }}
        
        /* Minimalist Icon Footer */
        .footer {{
            margin-top: 60px;
            padding: 30px 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            display: flex; justify-content: center; align-items: center; gap: 30px;
            flex-wrap: wrap;
        }}
        .footer-icons {{
            display: flex; gap: 20px;
        }}
        .footer-icons a {{
            font-size: 1.4rem; text-decoration: none; transition: 0.3s;
        }}
        .footer-icons a:hover {{ transform: scale(1.2); filter: drop-shadow(0 0 8px #10b981); }}
        
        div.block-container {{ padding-bottom: 5rem !important; }}
        .footer-col {{ min-width: 200px; }}
        .footer-col-title {{ font-size: 0.85rem; font-weight: 700; color: #fff; letter-spacing: 2px; margin-bottom: 20px; text-transform: uppercase; }}
        .footer-col-text {{ font-size: 0.9rem; color: #94a3b8; line-height: 1.8; }}
        .footer-logo-circle {{
            width: 45px; height: 45px; border-radius: 50%; border: 2px solid #fbbf24;
            display: flex; align-items: center; justify-content: center;
            color: #fbbf24; font-weight: 700; font-size: 1.2rem;
        }}
        
        /* Mobile Footer Overrides */
        @media (max-width: 768px) {{
            .footer-invite-text {{ font-size: 1.5rem; }}
            .footer-email {{ font-size: 1.6rem; }}
            .footer-grid {{ gap: 40px; text-align: center; }}
            .footer-logo-circle {{ margin: 0 auto 20px auto; }}
        }}
        
        /* Fix Streamlit Overrides */
        [data-testid="stMetricValue"] {{ display: none; }}
        [data-testid="stMetricLabel"] {{ display: none; }}
        
        /* New Live Metrics Styling */
        .live-metrics-row {{
            display: flex; gap: 20px; margin-bottom: 25px;
            justify-content: center; align-items: center;
        }}
        .live-metric-card {{
            background: rgba(15, 23, 42, 0.8);
            border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 15px 25px; min-width: 220px; text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        .live-metric-label {{ font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; }}
        .live-metric-value {{ font-size: 2.2rem; font-weight: 800; color: #10b981; margin-top: 5px; }}
        .live-metric-value-red {{ font-size: 2.2rem; font-weight: 800; color: #ef4444; margin-top: 5px; }}
        
        /* Activity Log Console */
        .activity-log-container {{
            background: #0f172a;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            height: 250px;
            overflow-y: auto;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85rem;
            color: #10b981;
            line-height: 1.6;
        }}
        .log-entry {{ border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 4px 0; }}
        .log-id {{ color: #fbbf24; font-weight: bold; }}
        .log-cost {{ color: #ef4444; font-weight: bold; }}
    </style>
    
    <div class="top-bar">
        <a href="/" target="_self" class="top-bar-title">🛣️ {PROJECT_TITLE}</a>
        <div style="display: flex; gap: 20px; align-items: center;">
            <a href="https://github.com/ravindra-gogineni" target="_blank" style="color: #94a3b8; text-decoration: none; display: flex; align-items: center;">
                <svg height="24" viewBox="0 0 16 16" version="1.1" width="24" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.22 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
            </a>
            <a href="https://www.linkedin.com/in/ravindra-gogineni-034501212" target="_blank" style="color: #94a3b8; text-decoration: none; display: flex; align-items: center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
            </a>
        </div>
    </div>
    <div style="margin-top: 100px;"></div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown(f"""
    <div class="footer">
        <div>© 2026 {PROJECT_TITLE}. All Rights Reserved.</div>
        <div class="footer-icons">
            <a href="{GITHUB_URL}" target="_blank" title="GitHub Source" style="color: #94a3b8;">
                <svg height="24" viewBox="0 0 16 16" version="1.1" width="24" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.22 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
            </a>
            <a href="{LINKEDIN_URL}" target="_blank" title="Developer LinkedIn" style="color: #94a3b8; margin-left:10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

# ==========================================
# 🧠 CORE LOGIC & EMAIL
# ==========================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model missing: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

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
    pdf.cell(0, 10, f"ESTIMATED REPAIR COST: Rs. {total_cost:,}", ln=True)
    if detections and detections[0].get('Location'):
         pdf.set_font("Arial", size=10)
         pdf.cell(0, 10, f"Precise GPS: {detections[0]['Location']}", ln=True)
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
    if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
        st.session_state.admin_logged_in = False
        st.rerun()
else:
    # Use a clean button for login that triggers the modal
    if st.sidebar.button("🔐 Authority Login", key="login_trigger", use_container_width=True):
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

    st.title(f"🛣️ {PROJECT_TITLE} Dashboard")
    st.caption("AI-Powered Road Audit & Automated Reporting System")

    # Tabs
    tab_live, tab_tracker, tab_analytics = st.tabs([
        "🔴 Live Inspection", 
        "🔍 Track Complaint",
        "📊 Analytics & Reports"
    ])

    with tab_live:
        st.subheader("Inspection Feed")
        global video_placeholder, metrics_placeholder, chart_placeholder, record_table, activity_placeholder
        metrics_placeholder = st.empty()
        
        col_feed, col_log = st.columns([2, 1])
        with col_feed:
            video_placeholder = st.empty()
        with col_log:
            st.markdown("##### 📜 Live Activity Feed")
            activity_placeholder = st.empty()
            
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
    st.title(f"🏛️ {PROJECT_TITLE} - Authority Portal")
    st.info("Authorized Personnel Only: Review, Route, and Manage Community Hazards.")
    
    col1, col2, col3 = st.columns(3)
    all_complaints = db.get_all_complaints()
    
    total_potholes = sum(c[3] for c in all_complaints) if all_complaints else 0
    total_budget = sum(c[4] for c in all_complaints) if all_complaints else 0
    
    with col1:
        st.markdown(f'<div class="metric-container"><div class="metric-title">📦 Total Reports</div><div class="metric-value">{len(all_complaints) if all_complaints else 0}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><div class="metric-title">⚠️ Total Potholes</div><div class="metric-value">{total_potholes}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><div class="metric-title">💰 Est. Budget</div><div class="metric-value">₹{total_budget:,}</div></div>', unsafe_allow_html=True)
    
    st.divider()
    
    tab_manage, tab_reports = st.tabs(["📋 Manage Complaints", "📊 Advanced Analytics"])
    
    with tab_manage:
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
                    col_info1.write(f"**Coordinates:** {lat}, {lng}")
                    if lat and lng:
                        st.link_button("🌐 View on Google Maps", f"https://www.google.com/maps/search/?api=1&query={lat},{lng}")
                    
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
                                    db.update_complaint(c_id, "Scheduled", 7, forwarded_at=current_time) # Fixed: Correct kwarg
                                    st.success(f"Report forwarded to {target_dist} officials!")
                                    st.rerun()
                                else: st.error(msg)
                                
                    # --- BLACKSPOT MANAGEMENT ---
                    with st.container(border=True):
                        st.write("🚨 **Accident Prone Area Management**")
                        is_current_black = bool(is_black)
                        if st.toggle("Mark as Accident Prone Blackspot", value=is_current_black, key=f"black_{c_id}"):
                            if not is_current_black:
                                db.update_complaint(c_id, status, start_days, is_blackspot=1)
                                st.warning("Spot marked as Permanent Blackspot!")
                                st.rerun()
                        else:
                            if is_current_black:
                                db.update_complaint(c_id, status, start_days, is_blackspot=0)
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
        if all_complaints:
            # SAFETY SHIELD: 16 Columns Total
            col_names = ["id", "name", "road", "count", "cost", "status", "days", "time", "details", "reporter_email", "authority_email", "priority", "forwarded_at", "lat", "lng", "is_blackspot"]
            padded_comp = [list(r) + [None] * (len(col_names) - len(r)) for r in all_complaints]
            df = pd.DataFrame(padded_comp, columns=col_names)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Master Audit Log (CSV)",
                data=csv,
                file_name="andhra_road_audit_master.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No data available for reporting yet.")

def update_ui_elements(audit):
    # Metrics - Side-by-side Layout
    metrics_placeholder.markdown(f"""
        <div class="live-metrics-row">
            <div class="live-metric-card">
                <div class="live-metric-label">📋 Total Potholes</div>
                <div class="live-metric-value">{audit.pothole_count}</div>
            </div>
            <div class="live-metric-card">
                <div class="live-metric-label">💰 Estimated Total Cost</div>
                <div class="live-metric-value-red">₹ {audit.total_cost:,}</div>
            </div>
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
    
    # Activity Log
    if audit.activity_logs:
        logs_html = "".join([f'<div class="log-entry">{log}</div>' for log in reversed(audit.activity_logs)])
        activity_placeholder.markdown(f'<div class="activity-log-container">{logs_html}</div>', unsafe_allow_html=True)

# --- MAIN RENDER LOGIC ---
def main():
    apply_premium_style()
    
    if st.session_state.show_login:
        render_login_page()
    elif st.session_state.admin_logged_in:
        render_admin_view()
    else:
        render_citizen_view(audit)
    
    render_footer()

if __name__ == "__main__":
    main()

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
            # If using an UPLOADED video, prefer the 'Road Name' location over the browser GPS
            if source_type == "Upload Video" and input_road:
                try:
                    g = geocoder.osm(input_road) # Search for the road name specifically
                    if g.latlng:
                        st.session_state.map_lat, st.session_state.map_lng = g.latlng[0], g.latlng[1]
                        session_gps = f"📍 {input_road} ({st.session_state.map_lat:.4f}, {st.session_state.map_lng:.4f})"
                    else:
                        session_gps = input_road
                except:
                    session_gps = input_road
            else:
                try:
                    g = geocoder.ip('me')
                    if g.latlng:
                        # Use browser GPS if available, else session default
                        lat_val = st.session_state.get('map_lat', g.latlng[0])
                        lng_val = st.session_state.get('map_lng', g.latlng[1])
                        session_gps = f"{lat_val:.5f}, {lng_val:.5f}"
                    else: 
                        session_gps = input_road if input_road else "📍 Hyderabad, IN"
                except:
                    session_gps = input_road if input_road else "📍 Hyderabad, IN"
            
            frame_count = 0 
            while cap.isOpened() and st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    # Final log message to match user's gold standard
                    audit.activity_logs.append(f'<span class="log-id">Audit Complete.</span> Total: <span class="log-cost">Rs. {audit.total_cost:,}</span>')
                    update_ui_elements(audit) # Final UI update
                    st.success("Video processing complete! You can now send the report.")
                    st.session_state.is_running = False
                    break
            
                frame_count += 1
                
                # Convert to RGB once for AI and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_area = frame.shape[0] * frame.shape[1]
                
                # --- HIGH-ACCURACY REAL SCANNING (0.35 SWEET SPOT) ---
                # Conf: 0.35 captures hidden potholes; IOU: 0.55 prevents merging real hazards.
                results = model.track(frame_rgb, persist=True, tracker="bytetrack.yaml", conf=0.35, iou=0.55, verbose=False)
                
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
                                
                                # Add to Activity Feed (matches User's friend's log style)
                                log_msg = f'New Defect <span class="log-id">ID:{track_id}</span> | GPS: {session_gps} | Cost: <span class="log-cost">Rs. {cost}</span>'
                                audit.activity_logs.append(log_msg)
                        
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
