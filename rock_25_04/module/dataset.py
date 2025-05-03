import os
import numpy as np
import cv2, random
from torch.utils.data import Dataset
from collections import Counter, defaultdict

class PlainDataset(Dataset):
    def __init__(self, data_path, matcher):
        self.data = []
        self.labels = []
        self.class_names = []

        for class_name in os.listdir(data_path):
            class_dir = os.path.join(data_path, class_name)
            if not os.path.isdir(class_dir): continue

            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                img = cv2.imread(fpath)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                self.data.append(img)  # ✅ 이미지 메모리 저장
                self.labels.append(matcher[class_name])
                self.class_names.append(class_name)

        self.data = np.array(self.data, dtype=object)
        self.labels = np.array(self.labels)
        self.class_names = np.array(self.class_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        return img, label


class TrainAugDataset(Dataset):
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

        self.data = origin_dataset.data
        self.labels = origin_dataset.labels
        self.class_names = origin_dataset.class_names

        self.augmented_data = []
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
                img = self.data[idx].copy()
                label = self.labels[idx]
                class_name = self.class_names[idx]

                transform_to_use = (
                    self.custom_transform if self.use_custom_condition_fn and self.use_custom_condition_fn(class_name)
                    else self.default_transform
                )
                if transform_to_use:
                    img = transform_to_use(img, mode='full')
                    if img is None:
                        continue

                self.augmented_data.append(img)
                self.augmented_labels.append(label)

        print("[After Augmentation]", Counter(np.concatenate([self.labels, self.augmented_labels])))

    def __len__(self):
        return len(self.data) + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < len(self.data):
            img = self.data[idx]
            label = self.labels[idx]

            class_name = self.class_names[idx]
            transform_to_use = (
                self.custom_transform if self.use_custom_condition_fn and self.use_custom_condition_fn(class_name)
                else self.default_transform
            )
            img = transform_to_use(img, mode='resize_only') if transform_to_use else img
        else:
            img = self.augmented_data[idx - len(self.data)]
            label = self.augmented_labels[idx - len(self.data)]

        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        return img, label
