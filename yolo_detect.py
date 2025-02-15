import os
import cv2
from ultralytics import YOLO
import torch
import multiprocessing
import glob

def main():
    project_dir = "./YOLO-AraOCR/runs/train"

    model = YOLO("yolo11n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cpu_cores = multiprocessing.cpu_count()
    workers = min(8, cpu_cores // 2)  # Use half the available cores (max 8)
    print(f"Using device: {device}")
    print(f"Using {workers} CPU workers for data loading")

    model.train(
        data="./YOLO-AraOCR/book_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=2,
        device=device,
        workers=workers,
        project=project_dir,
        val=True,
    )

if __name__ == "__main__":
    main()
