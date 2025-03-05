import os
import cv2
import albumentations as A
import numpy as np
from collections import Counter

# 원본 이미지 경로 & 저장할 증강 이미지 경로
if not os.getcwd().endswith('data'):
    os.chdir('./image_icon_25_03/data')
data_path = "train"
save_path = "train_data"  # 증강된 이미지 저장 경로
os.makedirs(save_path, exist_ok=True)

# 클래스 레이블 정의
labels = list(range(10))  # [0,1,2,3,4,5,6,7,8,9]

# 각 클래스별 파일 개수 확인
file_counts = {label: len([f for f in os.listdir(data_path) if f"_{label}.png" in f]) for label in labels} 

# 가장 많은 클래스 개수 찾기
max_count = max(file_counts.values())

# Albumentations 기반 이미지 증강 파이프라인
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
])

# 부족한 클래스 이미지 증강
for label, count in file_counts.items():
    if count < max_count:
        existing_images = [f for f in os.listdir(data_path) if f.endswith(f"_{label}.png")]
        additional_needed = max_count - count  # 부족한 개수

        for i in range(additional_needed):
            img_name = np.random.choice(existing_images)  # 랜덤 샘플 선택
            img_path = os.path.join(data_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 증강 수행
            augmented = augmentations(image=img)['image']

            # 기존 id 유지하고 새 파일명 생성
            original_id = img_name.split('_')[0]  # 기존 ID 추출
            new_filename = f"{original_id}_aug_{i}_{label}.png"  # 증강된 파일명 생성: {기존 ID}_aug_{증강 ID}_{클래스}.png
            save_img_path = os.path.join(save_path, new_filename)
            cv2.imwrite(save_img_path, augmented)

print("✅ Data augmentation complete. Augmented images saved in:", save_path)
