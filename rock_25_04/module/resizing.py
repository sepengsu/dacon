import cv2, os
from tqdm import tqdm
import numpy as np

def get_resize_thresholds(required_img_size: int):
    """
    주어진 최종 입력 이미지 크기(required_img_size)에 따라
    제거 기준(min_size), 패딩 기준(resize_thresh)을 계산합니다.

    기준:
    - min_size: 최종 크기의 약 30% (과도한 확대 방지)
    - resize_thresh: 최종 크기의 약 50% (padding 허용 한계)

    Args:
        required_img_size (int): 최종 resize될 이미지 크기 (ex: 384)

    Returns:
        dict: {
            "min_size": 너무 작아서 제거할 최소 크기,
            "resize_thresh": padding과 resize 기준 경계
        }
    """
    min_size = int(required_img_size * 0.3)       # 예: 384 * 0.3 = 115
    resize_thresh = int(required_img_size * 0.5)  # 예: 384 * 0.5 = 192

    return {
        "min_size": min_size,
        "resize_thresh": resize_thresh
    }

class CustomRockResize:
    def __init__(self, resize_size=384, resize_thresh=192, min_size=120):
        """
        CustomRockResizeCV2

        이 클래스는 OpenCV(cv2)를 활용해 건설용 자갈 이미지를 전처리합니다.  
        입력 이미지의 크기 편차가 매우 크기 때문에, 다음 세 가지 기준에 따라 처리합니다:

        1. 이미지의 너비 또는 높이가 너무 작으면(min_size 미만) 해당 이미지는 제외합니다.
        2. 이미지 크기가 중간 수준이면(resize_thresh 미만), 
        비율을 유지한 상태로 리사이즈한 후, 검정색(0)으로 패딩하여 최종 크기(resize_size x resize_size)에 맞춥니다.
        3. 이미지가 충분히 크면 비율만 유지하여 바로 resize합니다.

        이 방식은 모든 입력 이미지를 동일한 크기로 통일(예: 384x384)하면서도,  
        불필요한 왜곡이나 정보 손실을 최소화하도록 설계되어 있습니다.

        매우 작은 이미지의 품질 저하를 방지하고, CNN 및 ViT 기반 모델 학습을 위한 안정적인 전처리를 제공합니다.

        매개변수:
            resize_size (int): 최종 출력 이미지 크기 (기본값: 384)
            resize_thresh (int): 패딩과 리사이즈를 나누는 기준 크기 (기본값: 192)
            min_size (int): 허용하는 최소 이미지 크기 (기본값: 120)

        반환값:
            처리된 이미지 (np.ndarray, HWC 형식) 또는 너무 작을 경우 None
        """

        self.resize_size = resize_size          # 최종 출력 크기 (예: 384x384)
        self.resize_thresh = resize_thresh      # resize vs padding 기준 (예: 192)
        self.min_size = min_size                # 너무 작으면 제거 (예: 120)

    def __call__(self, img_np: np.ndarray):
        # [1] 이미지 크기 추출 (cv2: (H, W, C))
        h, w, _ = img_np.shape
        min_wh = min(w, h)

        # [2] 너무 작은 경우는 제외 (None 반환)
        if min_wh < self.min_size:
            return None

        # [3] 중간 크기 → 비율 유지 resize 후 padding to (384, 384)
        elif min_wh < self.resize_thresh:
            # ① 축소/확대 비율 계산 (최대 변 기준)
            scale = self.resize_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)

            # ② 비율 유지 resize
            img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # ③ 남은 공간만큼 padding 계산
            pad_w = self.resize_size - new_w
            pad_h = self.resize_size - new_h

            # ④ 검정색으로 패딩 (좌우, 상하 대칭)
            img_padded = cv2.copyMakeBorder(
                img_resized,
                pad_h // 2, pad_h - pad_h // 2,  # top, bottom
                pad_w // 2, pad_w - pad_w // 2,  # left, right
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # 검정색 (BGR)
            )
            return img_padded

        # [4] 충분히 크면 → 비율 유지 resize to (384, 384)
        else:
            scale = self.resize_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_np, (new_w, new_h))

            pad_w = self.resize_size - new_w
            pad_h = self.resize_size - new_h
            img_padded = cv2.copyMakeBorder(
                img_resized,
                pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
            return img_padded


class ResizeAndSaver:
    def __init__(self, input_dir, output_dir, resizer):
        """
        Args:
            input_dir (str): 원본 이미지 루트 디렉토리 (클래스별 서브폴더 포함)
            output_dir (str): 리사이즈 이미지 저장 루트 디렉토리
            resizer (CustomRockResize): 리사이즈 및 패딩 처리기
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.resizer = resizer

    def run(self, verbose=True):
        skipped, saved = 0, 0
        os.makedirs(self.output_dir, exist_ok=True)

        class_dirs = sorted(os.listdir(self.input_dir))
        for class_name in tqdm(class_dirs, desc="📁 Processing classes"):
            class_path = os.path.join(self.input_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            output_class_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                if not os.path.isfile(fpath):
                    continue

                img = cv2.imread(fpath)
                if img is None:
                    skipped += 1
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = self.resizer(img)

                if resized is None:
                    skipped += 1
                    continue

                # RGB -> BGR 다시 변환하여 저장
                save_path = os.path.join(output_class_dir, fname)
                cv2.imwrite(save_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                saved += 1

        if verbose:
            print(f"\n✅ Saved images: {saved}")
            print(f"⚠️ Skipped (too small or error): {skipped}")