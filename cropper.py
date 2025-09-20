import os
import yaml
import cv2
import numpy as np

skipped = 0

def resize_with_padding(image, target_size=(224, 224), pad_color=(0, 0, 0)):
    """Resize image keeping aspect ratio, pad to target size."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    if new_w < w or new_h < h:
        # for shrinking
        interpolation = cv2.INTER_AREA  
    else:
        # for enlarging
        interpolation = cv2.INTER_LINEAR  
        
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # create new image with padding
    result = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result


def crop_from_split(split_folder, split_name, class_lists, target_folder, output_size):
    global skipped
    print(f"Processing split: {split_name}")
    images_dir = os.path.join(split_folder, "images")
    labels_dir = os.path.join(split_folder, "labels")
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        
        label_path = os.path.join(labels_dir, label_file)
        base_name = os.path.splitext(label_file)[0]

        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_image_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                break
        else:
            print(f"Warning: Could not find image file for {base_name}")
            continue
        
        if not os.path.exists(image_path):
            continue 
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        # Ensure image is in 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w, _ = image.shape
        
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            try:
                cls_id, x_center, y_center, bw, bh = map(float, line.strip().split())
                cls_id = int(cls_id)
            except ValueError as e:
                skipped += 1
                print(f"Skipping line with invalid data in {label_path}: {line}. Error: {e}")
                continue

            class_name = class_lists[cls_id]
            
            # convert YOLO -> pixel coords
            x_center *= w
            y_center *= h
            bw *= w
            bh *= h
            
            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)
            
            crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size == 0:
                continue
            
            resized_crop = resize_with_padding(crop, target_size=output_size)

            out_dir = os.path.join(target_folder, split_name, class_name)
            os.makedirs(out_dir, exist_ok=True)
            
            out_name = f"{split_name}_{os.path.splitext(label_file)[0]}_symbol_{i}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, resized_crop)
            
            if i % 10 == 0: 
                print(f"Processing {label_file}, symbol {i}")

def file_finder(dir):
    dataset_config_file = None
    items = os.listdir(dir)
    train_folder = os.path.join(dir, "train")
    valid_folder = os.path.join(dir, "valid")
    test_folder = os.path.join(dir, "test")
    
    # find configuration file
    for item in items:
        full_path = os.path.join(dir, item)
        if item == "data.yaml" and os.path.isfile(full_path):
            dataset_config_file = full_path
            break
    
    # read classes from yaml
    if dataset_config_file:
        with open(dataset_config_file, "r") as f:
            data = yaml.safe_load(f)
            class_lists = data.get("names", [])
    
    return class_lists, train_folder, valid_folder, test_folder


def crop_labels(dir_dataset = None, output_size=(224, 224)):
    target_folder = "cropped-datasets"
    class_lists, train_folder, valid_folder, test_folder = file_finder(dir=dir_dataset)

    for split_folder, split_name in [(train_folder, "train"), (valid_folder, "valid"), (test_folder, "test")]:
        if os.path.exists(split_folder):
            crop_from_split(split_folder, split_name, class_lists, target_folder, output_size)
    
    print(f"\n\n Finished cropping. Skipped lines in labels: {skipped}")

if __name__ == "__main__":
    crop_labels(dir_dataset=r".\raw-datasets", output_size=(224, 224))
    