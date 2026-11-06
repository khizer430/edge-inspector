import cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    # Paths
    model_path = Path("runs/detect/train/weights/best.pt")
    video_in = Path("data/synthetic/run_001/preview.mp4")
    video_out = Path("data/synthetic/run_001/inference_output.mp4")

    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

    print(f"Processing video...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, verbose=False)
        
        # Plot bounding boxes on the frame
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Saved annotated video to {video_out} ({frame_count} frames)")

if __name__ == "__main__":
    main()