import os
import json
import random
import shutil
import sys
from glob import glob
from PIL import Image

def verify_and_restore_images(images_dir, dataset_root, labels_dir):
    for img_path in (glob(os.path.join(images_dir, "*.jpg")) +
                     glob(os.path.join(images_dir, "*.png")) +
                     glob(os.path.join(images_dir, "*.jpeg"))):
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify image integrity
        except Exception as e:
            print(f"Corrupt image detected: {img_path}. Attempting to restore...")
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # Convert to a safe format
                    restored_path = os.path.splitext(img_path)[0] + ".jpg"  # Save as JPG
                    img.save(restored_path, "JPEG")
                    if restored_path != img_path:
                        os.remove(img_path)
                print(f"Restored: {restored_path}")
            except Exception as restore_error:
                corrupted_dir = os.path.join(dataset_root, "corrupted")
                os.makedirs(corrupted_dir, exist_ok=True)
                print(f"Failed to restore {img_path}: {restore_error}")
                shutil.move(img_path, os.path.join(corrupted_dir, os.path.basename(img_path)))
                # Also move the corresponding label if it exists
                label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                if os.path.exists(label_path):
                    shutil.move(label_path, os.path.join(corrupted_dir, os.path.basename(label_path)))

def check_duplicates(directory, extensions):
    file_list = []
    for ext in extensions:
        file_list.extend([os.path.splitext(os.path.basename(f))[0] 
                          for f in glob(os.path.join(directory, f"*.{ext}"))])
    seen = set()
    duplicates = set()
    for file in file_list:
        if file in seen:
            duplicates.add(file)
        else:
            seen.add(file)
    for duplicate in duplicates:
        duplicate_files = [f for f in glob(os.path.join(directory, "*")) 
                           if os.path.splitext(os.path.basename(f))[0] == duplicate]
        if len(duplicate_files) > 1:
            print(f"Duplicate found: {duplicate}. Keeping one and deleting others...")
            for file in duplicate_files[1:]:
                os.remove(file)

def verify_missing_labels(images_dir, labels_dir, dataset_root):
    missing_labels = []
    image_files = sorted(glob(os.path.join(images_dir, "*.jpg")) +
                         glob(os.path.join(images_dir, "*.png")) +
                         glob(os.path.join(images_dir, "*.jpeg")))
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            missing_labels.append(img_name)
            img_no_label_dir = os.path.join(dataset_root, "img_no_label")
            os.makedirs(img_no_label_dir, exist_ok=True)
            shutil.move(img_path, os.path.join(img_no_label_dir, img_name))
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images are missing labels! Moved to {img_no_label_dir}.")
        with open(os.path.join(dataset_root, "missing_labels.log"), "w") as f:
            f.write("\n".join(missing_labels))

def verify_extra_labels(images_dir, labels_dir, dataset_root):
    label_files = sorted(glob(os.path.join(labels_dir, "*.txt")))
    extra_labels = []
    for label_path in label_files:
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        img_path_jpg = os.path.join(images_dir, label_name + ".jpg")
        img_path_png = os.path.join(images_dir, label_name + ".png")
        img_path_jpeg = os.path.join(images_dir, label_name + ".jpeg")
        if not (os.path.exists(img_path_jpg) or os.path.exists(img_path_png) or os.path.exists(img_path_jpeg)):
            extra_labels.append(label_path)
            label_no_img_dir = os.path.join(dataset_root, "label_no_img")
            os.makedirs(label_no_img_dir, exist_ok=True)
            shutil.move(label_path, os.path.join(label_no_img_dir, os.path.basename(label_path)))
    if extra_labels:
        print(f"WARNING: {len(extra_labels)} labels found without matching images! Moved to {label_no_img_dir}.")

def bbox_from_points(points, geometry_type):
    if "exterior" not in points or not points["exterior"]:
        return None
    polygon = points["exterior"]
    try:
        if geometry_type == "polygon":
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        elif geometry_type == "rectangle":
            if len(polygon) != 2:
                return None
            return polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1]
    except Exception as e:
        print(f"Error processing bounding box: {e}")
    return None

# Convert JSON annotations to YOLO format labels and extract class information
def process_json_files(json_folder, labels_dir, dataset_root):
    all_classes = {}
    for json_file in glob(os.path.join(json_folder, "*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data["objects"]:
            class_id = obj.get("classId")
            class_title = obj.get("classTitle")
            if class_id and class_title:
                all_classes[class_id] = class_title
    print("\nClasses:")
    for class_id, class_title in all_classes.items():
        print(f"{class_id} {class_title}")
    sorted_classes = sorted(all_classes.items(), key=lambda x: x[0])
    category_map = {class_id: i for i, (class_id, _) in enumerate(sorted_classes)}
    
    classes_file = os.path.join(dataset_root, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for _, name in sorted_classes:
            f.write(name + "\n")
    print(f"Classes extracted and saved successfully in: {classes_file}")

    missing_bboxes = []
    for json_file in glob(os.path.join(json_folder, "*.json")):
        file_name = os.path.basename(json_file)
        image_name = os.path.splitext(file_name)[0]
        label_file = os.path.join(labels_dir, f"{image_name}.txt")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
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
    if missing_bboxes:
        missing_log = os.path.join(dataset_root, "missing_bboxes.log")
        with open(missing_log, "w", encoding="utf-8") as f:
            f.write("\n".join(missing_bboxes))
        print("Some objects are missing bounding boxes! Check `missing_bboxes.log` for details.")
    print(f"\nAll JSON files converted! Labels saved in `{labels_dir}`")

# Return True if the directory is empty or does not exist
def check_empty_directory(directory):
    return not os.path.exists(directory) or len(os.listdir(directory)) == 0

def train_test_split(images_dir, labels_dir, dataset_root, train_ratio=0.8):
    train_img_dir = os.path.join(images_dir, "train")
    test_img_dir = os.path.join(images_dir, "test")
    train_label_dir = os.path.join(labels_dir, "train")
    test_label_dir = os.path.join(labels_dir, "test")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Check that target directories are empty
    for d in [train_img_dir, test_img_dir, train_label_dir, test_label_dir]:
        if not check_empty_directory(d):
            print(f"Error: Directory {d} is not empty. Please empty it before running this script to avoid duplicates.")
            sys.exit(1)

    image_files = sorted(glob(os.path.join(images_dir, "*.jpg"))+glob(os.path.join(images_dir, "*.jpeg"))+glob(os.path.join(images_dir, "*.png")))
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    def move_files(file_list, target_img_dir, target_label_dir):
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

    # Re-verify dataset integrity for both splits
    for split in ["train", "test"]:
        img_folder = os.path.join(images_dir, split)
        label_folder = os.path.join(labels_dir, split)
        image_files = sorted(glob(os.path.join(img_folder, "*.jpg"))+glob(os.path.join(img_folder, "*.jpeg"))+glob(os.path.join(img_folder, "*.png")))
        label_files = sorted(glob(os.path.join(label_folder, "*.txt")))
        missing_labels = []
        extra_labels = []
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_folder, label_name)
            if not os.path.exists(label_path):
                missing_labels.append(img_name)
        if missing_labels:
            print(f"ERROR: {len(missing_labels)} images in {split} have no labels!")
            print("Stopping script execution. Fix Train/Test script.")
            sys.exit(1)
        for label_path in label_files:
            label_name = os.path.splitext(os.path.basename(label_path))[0]
            if not any(os.path.exists(os.path.join(img_folder, label_name + ext)) 
                       for ext in [".jpg", ".jpeg", ".png"]):
                extra_labels.append(label_path)
        if extra_labels:
            print(f"ERROR: {len(extra_labels)} labels in {split} have no corresponding images!")
            print("Stopping script execution. Fix Train/Test script.")
            sys.exit(1)
    print("Dataset re-verification complete!\n")
    return train_img_dir, test_img_dir, train_label_dir, test_label_dir

def create_data_yaml(dataset_root, train_img_dir, test_img_dir):
    yaml_file = os.path.join(dataset_root, "data.yaml")
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

def main():
    dataset_root = "./YOLO-AraOCR/book_dataset"
    labels_dir = os.path.join(dataset_root, "labels")
    images_dir = os.path.join(dataset_root, "images")
    json_folder = os.path.join(dataset_root, "ann")
    os.makedirs(labels_dir, exist_ok=True)

    # Verify image integrity and restore any corrupted
    verify_and_restore_images(images_dir, dataset_root, labels_dir)
    
    # Check for duplicate images and labels
    check_duplicates(images_dir, ["jpg", "png", "jpeg"])
    check_duplicates(labels_dir, ["txt"])
    
    # Verify missing and extra labels
    verify_missing_labels(images_dir, labels_dir, dataset_root)
    verify_extra_labels(images_dir, labels_dir, dataset_root)
    
    # Process JSON files to generate YOLO labels and extract classes
    process_json_files(json_folder, labels_dir, dataset_root)
    
    # Train/Test split with empty-directory check
    train_img_dir, test_img_dir, train_label_dir, test_label_dir = train_test_split(
        images_dir, labels_dir, dataset_root, train_ratio=0.8)
    
    # Create data.yaml for YOLO training
    create_data_yaml(dataset_root, train_img_dir, test_img_dir)
    
    print("\nConversion complete! YOLO dataset is ready for training. 🚀\n")

if __name__ == "__main__":
    main()
