import shutil
from pathlib import Path
import random
import yaml

def main():
    # Paths
    run_dir = Path("data/synthetic/run_001")
    out_dir = Path("data/yolo_dataset")
    
    frames_dir = run_dir / "frames"
    labels_dir = run_dir / "yolo" / "labels"

    # Create YOLO folder structure
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get all frame files
    frames = sorted(list(frames_dir.glob("*.jpg")))
    
    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(frames)
    split_idx = int(len(frames) * 0.8)
    
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    def copy_files(frame_list, split_name):
        for f in frame_list:
            # Copy image
            shutil.copy(f, out_dir / "images" / split_name / f.name)
            # Copy corresponding label
            label_path = labels_dir / f.with_suffix(".txt").name
            if label_path.exists():
                shutil.copy(label_path, out_dir / "labels" / split_name / label_path.name)

    copy_files(train_frames, "train")
    copy_files(val_frames, "val")

    # Create the YOLO data.yaml file
    data_yaml = {
        "path": str(out_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "ok",
            1: "defect"
        }
    }

    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    print(f"[OK] Dataset prepared at {out_dir.absolute()}")
    print(f"   Train: {len(train_frames)} images")
    print(f"   Val:   {len(val_frames)} images")

if __name__ == "__main__":
    main()