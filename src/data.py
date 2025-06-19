import os
import shutil
import random
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

def balance_classes(input_dir, balanced_dir, target_count=None, copy_aug=True):
    print(f"[INFO] Balancing classes from: {input_dir}")
    os.makedirs(balanced_dir, exist_ok=True)

    class_counts = {}
    image_paths = defaultdict(list)

    #láy ảnh theo từng lớp(8 lớp)
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        class_counts[class_name] = len(images)
        for img in images:
            image_paths[class_name].append(os.path.join(class_path, img))

    #cân bằng dữ liệu 
    if target_count is None:
        target_count = int(input("Nhập số lượng: ")) #ví dự là 200 ảnh cho từng lớp
    print(f"[INFO] Ảnh cho từng lớp là: {target_count}")

    # cần bằng ảnh cho tùng lớp
    for class_name, paths in image_paths.items():
        class_out_dir = os.path.join(balanced_dir, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        current_count = len(paths)
        print(f"[INFO] Class '{class_name}': {current_count} -> {target_count}")
        if current_count >= target_count:
            selected = random.sample(paths, target_count)
            for src in selected:
                shutil.copy(src, os.path.join(class_out_dir, os.path.basename(src)))
        else:
            for src in paths:
                shutil.copy(src, os.path.join(class_out_dir, os.path.basename(src)))
            #ớp nào ít quá thì nhân bản thêm
            to_add = target_count - current_count
            for i in range(to_add):
                src = random.choice(paths)
                img = Image.open(src)
                #biến đổi nhẹ (ko ảnh hưởng tới dữ liệu)
                img = img.transpose(Image.FLIP_LEFT_RIGHT) if i % 2 == 0 else img.transpose(Image.FLIP_TOP_BOTTOM)
                filename = f"aug_{i}_{os.path.basename(src)}"
                img.save(os.path.join(class_out_dir, filename))

    print("[INFO] Hoàn thành cân bằng ảnh.\n")

#hàm sắp xếp dữ liệu :    train 70%, test 15%, thực tế: 15%
def split_dataset(balanced_dir, output_dir, test_size=0.15, val_size=0.15):
    print(f"[INFO] Dữ liệu lấy từ: {balanced_dir}")
    for class_name in os.listdir(balanced_dir):
        class_path = os.path.join(balanced_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        trainval, test = train_test_split(images, test_size=test_size, random_state=42)
        train, val = train_test_split(trainval, test_size=val_size / (1 - test_size), random_state=42)

        for split_name, split_list in zip(['train', 'val', 'test'], [train, val, test]):
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)
    
    print(f"[INFO] Sắp xếp dữ liệu tại: {output_dir}")


if __name__ == "__main__":
    raw_dir = r"E:\2025\PROJECT_AI\data"
    balanced_dir = r"E:\2025\PROJECT_AI\data_balanced"
    output_dir = r"E:\2025\PROJECT_AI\data_split"

    balance_classes(raw_dir, balanced_dir)
    split_dataset(balanced_dir, output_dir, test_size=0.15, val_size=0.15)
