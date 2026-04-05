import cv2
import time
import os
import math
import geocoder 
import numpy as np
import requests 
import base64   
from datetime import datetime
from ultralytics import YOLO
from fpdf import FPDF

# ==========================================
# ⚙️ USER CONFIGURATION
# ==========================================
# 1. AI MODEL PATH (Updated for Cloud/Root directory)
MODEL_PATH = "best.pt"

# 2. VIDEO SOURCE (Use 0 for Webcam, or 'demo.mp4' for file)
VIDEO_SOURCE = "demo.mp4"       
REPORT_FILENAME = f"Work_Order_{datetime.now().strftime('%Y%m%d')}.pdf"

# ==========================================
# 📧 EMAIL CONFIGURATION (BREVO API)
# ==========================================
# 1. Your API KEY (Get it from https://app.brevo.com/settings/keys/api)
# ⚠️ SECURITY: Never share this key on GitHub!
BREVO_API_KEY = "PASTE_YOUR_BREVO_KEY_HERE"

# 2. SENDER EMAIL 
# ⚠️ CRITICAL: Must be the ACTUAL email you used to sign up for Brevo.
# Do NOT use 'a0d65e001@smtp-brevo.com' or any other username.
SENDER_NAME = "Road Audit AI"
SENDER_EMAIL = "revathinalluri999@gmail.com"  # <--- PASTE YOUR REAL GMAIL HERE

# 3. RECIPIENT EMAILS (Who receives the report?)
RECEIVER_EMAILS = ["revathinalluri888@gmail.com", "22491a05z0@qiscet.edu.in","ravindra8022@gmail.com"]

# ==========================================
# 🧠 ROBUST LOGIC ENGINE
# ==========================================
class RoadAuditSystem:
    def __init__(self, model_path):
        print(f"Loading AI Model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Tracking variables
        self.processed_ids = set()      # Stores IDs we have already paid for
        self.processed_centroids = []   # Stores (x,y) of paid potholes to prevent duplicates
        self.detections = []            # List to store details for PDF
        self.total_project_cost = 0     
        self.pothole_count = 0

    def get_gps_location(self):
        """
        Returns the current GPS location.
        Default: Uses Wi-Fi/IP based location (Good for laptop testing).
        """
        try:
            # OPTION A: Wi-Fi / IP Location (Easiest for testing)
            g = geocoder.ip('me')
            if g.latlng:
                return f"{g.latlng[0]:.5f}, {g.latlng[1]:.5f}"
            
            # OPTION B: Real Hardware GPS (Uncomment if using USB GPS)
            # import serial, pynmea2
            # with serial.Serial('COM3', baudrate=9600, timeout=1) as ser:
            #     line = ser.readline().decode('utf-8')
            #     if line.startswith('$GNGGA'):
            #         msg = pynmea2.parse(line)
            #         return f"{msg.latitude:.5f}, {msg.longitude:.5f}"
            
            return "GPS Unavailable"
        except Exception:
            return "No Signal"

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
            color = (0, 255, 0) # Green
            extra_cost = percent_area * 50
            
        # 2. MEDIUM POTHOLE (Resurfacing)
        elif 2 <= percent_area < 8:
            severity = "MODERATE"
            color = (0, 165, 255) # Orange
            extra_cost = percent_area * 150
            
        # 3. LARGE POTHOLE (Full reconstruction)
        else:
            severity = "CRITICAL"
            color = (0, 0, 255) # Red
            extra_cost = percent_area * 300
            
        total_cost = int(base_cost + extra_cost)
        return severity, total_cost, color

    def is_duplicate_location(self, new_box):
        """Checks if a 'new' detection is actually just an old one flickering."""
        x1, y1, x2, y2 = new_box
        new_cx = (x1 + x2) // 2
        new_cy = (y1 + y2) // 2
        
        for (old_cx, old_cy) in self.processed_centroids:
            distance = math.sqrt((new_cx - old_cx)**2 + (new_cy - old_cy)**2)
            if distance < 50: # If within 50 pixels, it's the same pothole
                return True
        return False

    def process_video(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        print("Starting AI Road Audit (Press 'q' to stop)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_area = frame.shape[0] * frame.shape[1]

            # Run Tracking
            results = self.model.track(frame, persist=True, tracker="botsort.yaml", conf=0.4, verbose=False)

            # Draw Dashboard
            cv2.rectangle(frame, (0, 0), (380, 100), (0, 0, 0), -1)
            cv2.putText(frame, "PROJECT DASHBOARD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Potholes Billed: {self.pothole_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Total Est: Rs. {self.total_project_cost:,}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    severity, cost, color = self.calculate_severity(box, frame_area)
                    
                    if track_id not in self.processed_ids:
                        if not self.is_duplicate_location(box):
                            # NEW VALID DETCTION
                            self.processed_ids.add(track_id)
                            
                            # Save location
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            self.processed_centroids.append((cx, cy))
                            
                            # Get GPS
                            current_gps = self.get_gps_location()
                            
                            self.pothole_count += 1
                            self.total_project_cost += cost
                            
                            # Add to PDF list
                            self.detections.append({
                                "id": track_id,
                                "severity": severity,
                                "cost": f"Rs. {cost}",
                                "gps": current_gps,
                                "image": frame.copy()
                            })
                            print(f"New Defect ID:{track_id} | GPS: {current_gps} | Cost: Rs. {cost}")
                    
                    # Visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} | Rs.{cost}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("AI Road Audit System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Audit Complete. Total: Rs. {self.total_project_cost}")

    def generate_pdf(self):
        if not self.detections:
            print("No data to report.")
            return False

        print("Generating PDF...")
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "MUNICIPAL ROAD AUDIT REPORT", ln=True, align='C')
        pdf.ln(10)
        
        # Summary
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.cell(0, 10, f"Total Potholes: {self.pothole_count}", ln=True)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"TOTAL ESTIMATED COST: Rs. {self.total_project_cost:,}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)

        # List Detections
        for det in self.detections[:10]:
            pdf.set_fill_color(240, 240, 240)
            # Added GPS to the header line
            header_text = f"ID #{det['id']} [{det['severity']}] - Cost: {det['cost']}"
            pdf.cell(0, 10, header_text, ln=True, fill=True)
            
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 8, f"GPS Location: {det['gps']}", ln=True)
            pdf.set_font("Arial", size=12)
            
            # Save temp image
            img_path = f"temp_{det['id']}.jpg"
            cv2.imwrite(img_path, det['image'])
            pdf.image(img_path, x=10, w=100)
            pdf.ln(60)
            os.remove(img_path)

        pdf.output(REPORT_FILENAME)
        print(f"PDF Saved: {REPORT_FILENAME}")
        return True

    def send_notification(self):
        """Sends email using Brevo API (No SMTP/Port blocking issues)"""
        if not os.path.exists(REPORT_FILENAME):
            print("No report file to send.")
            return

        print(f"Sending email via Brevo API from {SENDER_EMAIL}...")
        
        try:
            # 1. Read and Encode the PDF
            with open(REPORT_FILENAME, "rb") as f:
                pdf_content = f.read()
                pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')

            # 2. Prepare the API URL and Headers
            url = "https://api.brevo.com/v3/smtp/email"
            headers = {
                "accept": "application/json",
                "api-key": BREVO_API_KEY,
                "content-type": "application/json"
            }

            # 3. Create the Payload
            payload = {
                "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
                "to": [{"email": email} for email in RECEIVER_EMAILS],
                "subject": f"URGENT: Road Audit Report - {datetime.now().strftime('%Y-%m-%d')}",
                "htmlContent": "<p>Please find attached the automated road audit report.</p>",
                "attachment": [
                    {
                        "content": pdf_base64,
                        "name": REPORT_FILENAME
                    }
                ]
            }

            # 4. Send Request
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 201:
                print("✅ Email Sent Successfully (via Brevo API)!")
            else:
                print(f"❌ Email Failed. Code: {response.status_code}")
                print(f"Reason: {response.text}")
                
        except Exception as e:
            print(f"Connection Error: {e}")

# ==========================================
# 🚀 RUN
# ==========================================
if __name__ == "__main__":
    app = RoadAuditSystem(MODEL_PATH)
    
    # 1. Run Detection
    app.process_video(VIDEO_SOURCE)
    
    # 2. Generate Report
    if app.generate_pdf():
        # 3. Send Email
        app.send_notification()