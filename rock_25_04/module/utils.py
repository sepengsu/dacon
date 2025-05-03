
import numpy as np
import torch, cv2
from collections import Counter, defaultdict

def compute_class_stats(dataset):
    """
    주어진 dataset에서 클래스별 샘플 수를 계산합니다.
    """
    label_counts = Counter(dataset.labels)
    return dict(sorted(label_counts.items()))

def compute_augmentation_ratios(label_counts, target_count='median', max_ratio=5.0):
    """
    클래스별 증강 비율을 자동 계산합니다.
    - target_count: 'median', 'mean', or int 지정 가능
    """
    counts = np.array(list(label_counts.values()))

    if target_count == 'median':
        target = int(np.median(counts))
    elif target_count == 'mean':
        target = int(np.mean(counts))
    elif isinstance(target_count, int):
        target = target_count
    else:
        raise ValueError("Invalid target_count")

    aug_ratios = {}
    for cls, count in label_counts.items():
        ratio = target / count
        aug_ratios[cls] = min(round(ratio, 1), max_ratio) if ratio > 1 else 1.0
    return aug_ratios, target

def apply_augmentation(dataset, aug_ratios, augmenter_fn=None):
    """
    증강 비율에 따라 샘플을 복제 + 증강하여 dataset을 확장합니다.
    - augmenter_fn: img, label -> 증강된 이미지
    """
    new_data = list(dataset.data)
    new_labels = list(dataset.labels)

    classwise_indices = defaultdict(list)
    for i, label in enumerate(dataset.labels):
        classwise_indices[label].append(i)

    for cls, ratio in aug_ratios.items():
        if ratio <= 1.0:
            continue

        indices = classwise_indices[cls]
        num_original = len(indices)
        num_additional = int((ratio - 1) * num_original)

        for _ in range(num_additional):
            idx = np.random.choice(indices)
            img_path = dataset.data[idx]
            label = dataset.labels[idx]

            # 실제 이미지 증강
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if augmenter_fn:
                img = augmenter_fn(img)
            # 저장하지 않고, 메모리에 경로로 유지
            new_data.append(img_path)
            new_labels.append(label)

    dataset.data = np.array(new_data)
    dataset.labels = np.array(new_labels)

    return dataset

def compute_class_weights(labels, num_classes):
    """
    주어진 라벨 배열(labels)로부터 class weight를 계산합니다.
    
    Args:
        labels (array-like): 라벨 목록 (list, np.array, 또는 torch.Tensor)
        num_classes (int): 전체 클래스 수

    Returns:
        torch.FloatTensor: 클래스별 가중치 (GPU로 이동)
    """
    label_counts = Counter(labels)
    counts = np.array([label_counts.get(i, 0) for i in range(num_classes)])

    # 0으로 나눔 방지
    counts[counts == 0] = 1

    weights = sum(counts) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to('cuda')