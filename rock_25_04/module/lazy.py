import os, cv2, random
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict, Counter

from torch.utils.data import Subset

class LazySubsetAdapter(Dataset):
    def __init__(self, base_dataset, indices):
        """
        LazyPlainDataset에 Subset 기능을 제공하는 어댑터

        Args:
            base_dataset (LazyPlainDataset): 전체 데이터셋
            indices (list or np.ndarray): 선택된 인덱스 리스트
        """
        self.base_dataset = base_dataset
        self.indices = indices

        # 주요 속성들도 부분적으로 추출
        self.img_paths = base_dataset.img_paths[indices]
        self.labels = base_dataset.labels[indices]
        self.class_names = np.array([
            os.path.basename(os.path.dirname(path)) for path in self.img_paths
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        return self.base_dataset[true_idx]



class LazyPlainDataset(Dataset):
    def __init__(self, data_path, matcher, return_path=False):
        self.img_paths = []
        self.labels = []
        self.class_names = []
        self.return_path = return_path

        for class_name in os.listdir(data_path):
            class_dir = os.path.join(data_path, class_name)
            if not os.path.isdir(class_dir): continue

            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath): continue

                self.img_paths.append(fpath)
                self.labels.append(matcher[class_name])
                self.class_names.append(class_name)

        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)
        self.class_names = np.array(self.class_names)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        fpath = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(fpath)
        if img is None:
            raise ValueError(f"[ERROR] Cannot read image: {fpath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0

        return (img, label, fpath) if self.return_path else (img, label)


class LazyTrainAugDataset(Dataset):
    def __init__(self, origin_dataset,
                 default_transform=None,
                 custom_transform=None,
                 use_custom_condition_fn=None,
                 target_count='median',
                 max_ratio=3.0):
        self.origin_dataset = origin_dataset
        self.default_transform = default_transform
        self.custom_transform = custom_transform
        self.use_custom_condition_fn = use_custom_condition_fn

        self.img_paths = origin_dataset.img_paths
        self.labels = origin_dataset.labels
        self.class_names = getattr(origin_dataset, 'class_names', np.array([
            os.path.basename(os.path.dirname(path)) for path in self.img_paths
        ]))

        self.augmented_paths = []
        self.augmented_labels = []

        self._augment(target_count, max_ratio)
        self.full_labels = np.concatenate([self.labels, self.augmented_labels])

    def _augment(self, target_count='median', max_ratio=3.0):
        label_counts = Counter(self.labels)
        print("[Before Augmentation]", dict(label_counts))

        counts = np.array(list(label_counts.values()))
        target = int(np.median(counts)) if target_count == 'median' else (
                 int(np.mean(counts)) if target_count == 'mean' else int(target_count))

        classwise_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            classwise_indices[label].append(i)

        for cls, count in label_counts.items():
            ratio = target / count
            if ratio <= 1.0:
                continue
            ratio = min(round(ratio, 1), max_ratio)
            num_additional = int((ratio - 1) * count)

            for _ in range(num_additional):
                idx = random.choice(classwise_indices[cls])
                self.augmented_paths.append(self.img_paths[idx])
                self.augmented_labels.append(self.labels[idx])

        print("[After Augmentation]", Counter(np.concatenate([self.labels, self.augmented_labels])))

    def __len__(self):
        return len(self.img_paths) + len(self.augmented_paths)

    def __getitem__(self, idx):
        if idx < len(self.img_paths):
            img_path = self.img_paths[idx]
            label = self.labels[idx]
            class_name = self.class_names[idx]
            mode = 'resize_only'
        else:
            aug_idx = idx - len(self.img_paths)
            img_path = self.augmented_paths[aug_idx]
            label = self.augmented_labels[aug_idx]
            class_name = os.path.basename(os.path.dirname(img_path))
            mode = 'full'

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise IOError
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            raise ValueError(f"[ERROR] Failed to load image: {img_path}")

        transform_to_use = (
            self.custom_transform if self.use_custom_condition_fn and self.use_custom_condition_fn(class_name)
            else self.default_transform
        )
        if transform_to_use:
            img = transform_to_use(img, mode=mode)
            if img is None:
                raise ValueError(f"[ERROR] Transform returned None: {img_path}")

        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        return img, label
