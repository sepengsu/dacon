📁 이미지 파일들 (클래스별 디렉토리 구조)
        │
        ▼
🧩 [Step 1] DualTransformDataset
        └─ 클래스 이름 기준으로 transform 분기
            ├─ if class_name == "Etc" → strong_transform_for_etc_wrapped
            └─ else → default_transform_wrapped
                    │
                    ▼
🧩 [Step 2] CustomRockResize
        ├─ min_size 미만 → None (제외됨)
        ├─ resize_thresh 미만 → 비율 유지 후 패딩
        └─ 그 외 → 비율 유지 resize (384x384)
                    │
                    ▼
🧪 [Step 3] 클래스별 transform
        ├─ Etc 클래스:
        │   ├─ Flip (H/V)
        │   ├─ Rotation ±20°
        │   ├─ Blur
        │   ├─ Brightness/Contrast 조절
        │   └─ Gaussian Noise 추가
        │
        └─ 기타 클래스:
            ├─ Flip (H)
            └─ Rotation ±10°
                    │
                    ▼
🧩 [Step 4] CHW 변환 + Normalize (0~1)
        ↓
🚚 DataLoader에 전달
        ↓
🧠 모델 학습 시작
