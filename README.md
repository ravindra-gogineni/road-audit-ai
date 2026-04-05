# 🛣️ Citizen Road Reporter: AI-Powered Road Audit SaaS

A professional, production-ready computer vision platform designed to automate road maintenance audits. Using **YOLOv8** and **Streamlit**, this dashboard allows citizens and authorities to detect, categorize, and invoice road defects (like potholes) in real-time.

## 🚀 Key Features
- **🤖 Real-Time AI Detection**: Uses a custom-trained YOLOv8 model to identify road defects with high precision.
- **📊 Live Analytics Dashboard**: Dynamic Plotly charts show defect severity distribution (Minor vs. Critical) as you scan.
- **📧 Automated Invoicing**: Instantly generates professional PDF reports with GPS tags and estimated repair costs, then dispatches them directly to road authorities.
- **🌍 Device Agnostic**: Fully responsive UI built for Mobile, Tablet, and Desktop users.
- **📥 Data Portability**: Export your audit results in **CSV** or **PDF** format with a single click.

## 🛠️ Tech Stack
- **Core**: Python 3.x
- **Framework**: Streamlit (SaaS Dashboard)
- **Computer Vision**: Ultralytics YOLOv8, OpenCV
- **Reporting**: FPDF (PDF Generation), Pandas (Data Management)
- **API Integration**: Brevo (Automated Email Dispatch)
- **Mapping**: Geocoder (GPS Integration)

## 📦 Quick Installation
To run this project locally, ensure you have Python 3.8+ installed and run:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

## 🛡️ Security
This project uses **GitHub Secrets** for API keys to ensure secure delivery of reports in a production environment.

---
*Developed for professional Road Maintenance Auditing and Citizen Reporting.*
