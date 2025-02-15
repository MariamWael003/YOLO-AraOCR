import os
import glob
import easyocr
from tqdm import tqdm

def ocr_pipeline(image_folder):
    reader = easyocr.Reader(['ar'], gpu=False)
    image_extensions = ["png", "jpg", "jpeg"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, f"*.{ext}")))
    total_imgs = len(image_files)
    print(f"Found {len(image_files)} images in {image_folder}")

    txt_dir = "./book_dataset/ocr_txts"
    os.makedirs(txt_dir, exist_ok=True)

    processed_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            results = reader.readtext(image_path, detail=0)
            if not results:
                tqdm.write(f"Warning: No text found in image {os.path.basename(image_path)}")
                continue

            recognized_text = "\n".join(results)
            
            base_name, _ = os.path.splitext(os.path.basename(image_path))
            txt_filename = f"{base_name}.txt"
            txt_path = os.path.join(txt_dir, txt_filename)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(recognized_text)

            processed_count += 1
        except Exception as e:
            tqdm.write(f"Error processing {os.path.basename(image_path)}: {e}")

    tqdm.write(f"Complete! Processed {processed_count}/{total_imgs}")

def main():
    cropped_images_folder = "./book_dataset/cropped_outputs"
    ocr_pipeline(cropped_images_folder)

if __name__ == "__main__":
    main()
