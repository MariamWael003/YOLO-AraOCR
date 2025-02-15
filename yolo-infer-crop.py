import os
import cv2
import glob
from ultralytics import YOLO

def infer_crop(model, test_folder, output_dir):
    image_files = []
    for ext in ["png", "jpg", "jpeg"]:
        image_files.extend(glob.glob(os.path.join(test_folder, f"*.{ext}")))
    print(f"Found {len(image_files)} images in {test_folder}")

    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        base_name, ext = os.path.splitext(os.path.basename(image_path))

        results = model(image)

        found_title = False
        found_text = False

        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates: [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                label = model.names[class_id] if hasattr(model, "names") else str(class_id)

                if label.lower() == "page":
                    continue

                # Process Title detections
                if label.lower() == "title":
                    found_title = True
                    cropped = image[y1:y2, x1:x2]
                    if cropped.size == 0:
                        print(f"Warning: Title crop is empty for image {image_path}")
                        continue
                    output_filename = f"{base_name}title{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    # If a file with the name exists, append an index
                    idx = 1
                    while os.path.exists(output_path):
                        output_filename = f"{base_name}title_{idx}{ext}"
                        output_path = os.path.join(output_dir, output_filename)
                        idx += 1
                    cv2.imwrite(output_path, cropped)
                    # print(f"Saved Title crop to {output_path}")

                # Process Body Text detections.
                elif label.lower() == "body text":
                    found_text = True
                    cropped = image[y1:y2, x1:x2]
                    if cropped.size == 0:
                        print(f"Warning: Body text crop is empty for image {image_path}")
                        continue
                    output_filename = f"{base_name}text{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    idx = 1
                    while os.path.exists(output_path):
                        output_filename = f"{base_name}text_{idx}{ext}"
                        output_path = os.path.join(output_dir, output_filename)
                        idx += 1
                    cv2.imwrite(output_path, cropped)
                    # print(f"Saved Body text crop to {output_path}")

                else:
                    print(f"Unknown label '{label}' in image {image_path}")

        if not found_title:
            print(f"No Title detection found for image {image_path}")
        if not found_text:
            print(f"No Body text detection found for image {image_path}")

def main():
    model_path = "./runs/train/train5/train/weights/best.pt"
    test_folder = "./book_dataset/images/test"
    output_dir = "./book_dataset/cropped_outputs"

    print("Loading model from", model_path)
    model = YOLO(model_path)
    print("Model loaded. Detected classes:", model.names)

    infer_crop(model, test_folder, output_dir)
    print("Cropping complete.")

if __name__ == "__main__":
    main()
