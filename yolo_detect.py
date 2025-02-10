import os
import cv2
from ultralytics import YOLO
import torch
import multiprocessing
import glob
import re

# Find the next available folder name with auto-incrementing numbers.
def get_next_folder(base_dir, prefix):
    existing_folders = sorted(glob.glob(os.path.join(base_dir, f"{prefix}*")))
    existing_indices = [
        int(folder.split(prefix)[-1]) for folder in existing_folders if folder.split(prefix)[-1].isdigit()
    ]
    next_index = max(existing_indices) + 1 if existing_indices else 1  # Always increments, never reuses numbers
    return os.path.join(base_dir, f"{prefix}{next_index}")

# Natural Sorting for Training Folders
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Define Training and Validation Directories (Auto-Incrementing)
project_dir = get_next_folder("runs/train", "train")
val_best_dir = get_next_folder("runs/val_best", "best_model")

model = YOLO("yolo11x.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

cpu_cores = multiprocessing.cpu_count()
workers = min(8, cpu_cores // 2)  # Use half the available cores (max 8)
print(f"Using device: {device}")
print(f"Using {workers} CPU workers for data loading")

model.train(
    data="/home/mariam/ML-Tech-Task/book_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=device,
    workers=workers,
    project=project_dir,
    val=True,
)

# Find the Best Model After Training
train_folders = sorted(glob.glob("runs/train/train*"), key=natural_sort_key, reverse=True)
if train_folders:
    latest_train_folder = train_folders[0]  # Get latest training run
    best_model_path = os.path.join(latest_train_folder, "weights", "best.pt")

    if os.path.exists(best_model_path):
        print(f"\nLoading Best Model: {best_model_path}")

        # Run validation separately for `best.pt`
        model = YOLO(best_model_path)
        metrics = model.val(project=val_best_dir, name="best_model")
        print("\nFinal Evaluation Metrics for Best Model:", metrics)

    else:
        print("\nError: Best model not found! Ensure training completed successfully.")
else:
    print("\nError: No training runs found!")

# Display the Confusion Matrix for Best Model
conf_matrix_path = os.path.join(val_best_dir, "best_model", "confusion_matrix.png")
if os.path.exists(conf_matrix_path):
    print("\nDisplaying Confusion Matrix for Best Model...")
    conf_img = cv2.imread(conf_matrix_path)
    cv2.imshow("Confusion Matrix (Best Model)", conf_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("\nWarning: Confusion matrix not found for best.pt. Ensure validation ran successfully.")

# Display Validation Images from Best Model
val_images_path = os.path.join(val_best_dir, "best_model", "predict")
if os.path.exists(val_images_path):
    print("\nDisplaying Validation Results for Best Model...")
    image_files = []
    for ext in ["jpg", "jpeg", "png"]:
        image_files.extend(glob.glob(os.path.join(val_images_path, f"*.{ext}")))
    if image_files:
        for img_path in image_files:
            img = cv2.imread(img_path)
            print("\nPress 'q' to exit, or 'Enter' to see the next image.")
            cv2.imshow("YOLO Validation Output (Best Model)", img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  
                break

        cv2.destroyAllWindows()
        print("\nValidation Output for Best Model Displayed!")
    else:
        print("\nWarning: No validation images found for best.pt.")
else:
    print("\nError: Validation results directory not found for best.pt. Skipping visualization.")
