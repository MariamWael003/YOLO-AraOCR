import os
import json
from glob import glob
import random
import shutil
import sys

dataset_root = "/home/mariam/ML-Tech-Task/book_dataset"
labels_dir = os.path.join(dataset_root, "labels")
images_dir = os.path.join(dataset_root, "images")
classes_file = os.path.join(dataset_root, "classes.txt")
json_folder = "/home/mariam/ML-Tech-Task/book_dataset/ann"
os.makedirs(labels_dir, exist_ok=True)

all_classes = {}

for json_file in glob(os.path.join(json_folder, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nProcessing JSON file: {json_file}...")

    for obj in data["objects"]:
        class_id = obj.get("classId")
        class_title = obj.get("classTitle")
        if class_id and class_title:
            all_classes[class_id] = class_title  # Store classId -> classTitle
            # print(f"Found class: {class_id} → {class_title}")  # Debugging

print("\nClasses:")
for class_id, class_title in all_classes.items():
    print(f"{class_id} {class_title}")
sorted_classes = sorted(all_classes.items(), key=lambda x: x[0])  # Sort by classId
category_map = {class_id: i for i, (class_id, _) in enumerate(sorted_classes)}

with open(classes_file, "w", encoding="utf-8") as f:
    for _, name in sorted_classes:
        f.write(name + "\n")

print(f"Classes extracted and saved successfully in: {classes_file}")

def bbox_from_points(points, geometry_type):
    if "exterior" not in points or not points["exterior"]:
        return None  # Return None if no exterior points exist

    polygon = points["exterior"]

    try:
        if geometry_type == "polygon":
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

        elif geometry_type == "rectangle":
            if len(polygon) != 2:
                return None  # Rectangles must have exactly two points
            return polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1]

    except Exception as e:
        print(f"Error processing bounding box: {e}")

    return None  # Return None if an error occurs

# Track missing bounding boxes
missing_bboxes = []

for json_file in glob(os.path.join(json_folder, "*.json")):
    file_name = os.path.basename(json_file)
    image_name = os.path.splitext(file_name)[0]  # Remove ".json" to get image ID
    label_file = os.path.join(labels_dir, f"{image_name}.txt")  # YOLO label file

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if image size exists
    if "size" not in data:
        print(f"Warning: Missing image size for {image_name}, skipping...")
        continue

    img_width = data["size"].get("width", 1)
    img_height = data["size"].get("height", 1)

    label_data = []

    for obj in data.get("objects", []):
        object_id = obj.get("id", "Unknown ID")
        class_id = obj.get("classId")
        class_title = obj.get("classTitle", "Unknown Class")
        geometry_type = obj.get("geometryType", "Unknown")
        points = obj.get("points", {})

        bbox = bbox_from_points(points, geometry_type)

        if bbox and class_id in category_map:
            x_min, y_min, x_max, y_max = bbox

            # Convert BBox to YOLO format
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            yolo_class_id = category_map[class_id]
            label_data.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            missing_bboxes.append(f"{image_name}: {class_title} (ID: {object_id}) has no bounding box")

    if label_data:
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(label_data))
        # print(f"YOLO labels saved for {image_name} -> {label_file}")

# Save missing bounding box log
if missing_bboxes:
    with open(os.path.join(dataset_root, "missing_bboxes.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(missing_bboxes))
    print("Some objects are missing bounding boxes! Check `missing_bboxes.log` for details.")

print(f"\nAll JSON files converted! Labels saved in `{labels_dir}`")

#Train/Test split
train_ratio = 0.8

train_img_dir = os.path.join(images_dir, "train")
test_img_dir = os.path.join(images_dir, "test")
train_label_dir = os.path.join(labels_dir, "train")
test_label_dir = os.path.join(labels_dir, "test")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

image_files = sorted(glob(os.path.join(images_dir, "*.jpg"))+glob(os.path.join(images_dir, "*.jpeg"))+glob(os.path.join(images_dir, "*.png")))

random.shuffle(image_files)
split_index = int(len(image_files) * train_ratio)
train_files = image_files[:split_index]
test_files = image_files[split_index:]

def move_files(file_list, target_img_dir, target_label_dir):
    # print(f"\nMoving {len(file_list)} files to {target_img_dir} and {target_label_dir}...")
    for img_path in file_list:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"

        src_label_path = os.path.join(labels_dir, label_name)
        target_label_path = os.path.join(target_label_dir, label_name)

        shutil.move(img_path, os.path.join(target_img_dir, img_name))

        if os.path.exists(src_label_path):
            shutil.move(src_label_path, target_label_path)
        else:
            print(f"Warning: Missing label for {img_name}")

move_files(train_files, train_img_dir, train_label_dir)
move_files(test_files, test_img_dir, test_label_dir)

print(f"Train/Test split completed! {len(train_files)} train / {len(test_files)} test")

# Verify dataset: Check if every image has a corresponding label
print("\nVerifying dataset integrity...")

for split in ["train", "test"]:
    img_folder = os.path.join(images_dir, split)
    label_folder = os.path.join(labels_dir, split)

    image_files = sorted(glob(os.path.join(img_folder, "*.jpg"))+glob(os.path.join(img_folder, "*.jpeg"))+glob(os.path.join(img_folder, "*.png")))

    missing_labels = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_folder, label_name)

        if not os.path.exists(label_path):
            missing_labels.append(img_name)

    if missing_labels:
        print(f"ERROR: {len(missing_labels)} images in {split} have no labels!")
        print("Stopping script execution. Please fix the dataset.")
        sys.exit(1)  # Stop script immediately

    else:
        print(f"All images in {split} have corresponding labels!")

print("Dataset verification complete!\n")

# Create data.yaml file
yaml_file = os.path.join(dataset_root, "data.yaml")

# Read class names from classes.txt
classes_file = os.path.join(dataset_root, "classes.txt")
with open(classes_file, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]
num_classes = len(class_names)

yaml_content = f"""# YOLOv11 Dataset Configuration
train: {train_img_dir}
val: {test_img_dir}  # Using test set for validation
nc: {num_classes}
names: {class_names}
"""

with open(yaml_file, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"data.yaml created successfully!")

print("\nConversion complete! YOLO dataset is ready for training. 🚀\n")

