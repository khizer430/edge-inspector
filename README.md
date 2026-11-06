# 🏭 Edge Inspector: Real-Time Visual Quality Control

An edge-deployed, full-stack AI system for detecting defects on a moving conveyor belt. Designed for manufacturing environments, this pipeline combines synthetic data generation, custom object detection, and a microservice architecture.

## 🏗️ Architecture
1. **Data Generator:** Procedurally generates a conveyor belt video with perfect ground-truth bounding boxes (YOLO format).
2. **Model:** Custom-trained YOLOv8n for detecting `ok` and `defect` metal castings.
3. **Backend:** FastAPI server for real-time image inference.
4. **Frontend:** Streamlit dashboard for monitoring, visual QC, and Accept/Reject telemetry.

## 🚀 Quickstart

**1. Install dependencies:**
```bash
pip install -r requirements.txt
