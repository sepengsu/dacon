import cv2
import numpy as np

def strong_transform_for_etc(img, resize_size=384):
    # Resize
    img = cv2.resize(img, (resize_size, resize_size))

    # Random Horizontal Flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    # Random Vertical Flip
    if np.random.rand() < 0.3:
        img = cv2.flip(img, 0)

    # Random Rotation ±20도
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-20, 20)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Gaussian Blur
    if np.random.rand() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1)

    # Color jitter: brightness & contrast
    if np.random.rand() < 0.4:
        alpha = np.random.uniform(0.8, 1.2)  # contrast
        beta = np.random.uniform(-15, 15)    # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Add Gaussian noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def default_transform(img, resize_size=384):
    # Resize
    img = cv2.resize(img, (resize_size, resize_size))

    # Horizontal Flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    # Small Rotation ±10도
    if np.random.rand() < 0.3:
        angle = np.random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return img


def wrapper(resizer, transform, img_size=384):
    def _wrapped(img, mode='full'):
        img = resizer(img)
        if img is None:
            return None
        if mode == 'resize_only':
            return img
        return transform(img, img_size)
    return _wrapped