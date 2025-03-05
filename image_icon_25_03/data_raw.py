import pandas as pd
import numpy as np
import cv2, os, json

# 데이터 분포 1024-> 32x32로 변환
def split_reshape(df):
    if 'label' not in df.columns: 
        ids = df['ID']
        images = df.drop(columns=['ID'])
        images = images.values.reshape(-1, 32, 32)
        print(images.shape) # (1024, 1024)
        return ids, images, None
    ids, labels =df['ID'], df['label']
    images = df.drop(columns=['ID', 'label']) # ID, label 제외한 나머지 데이터
    images = images.values.reshape(-1, 32, 32) # 32x32로 변환
    print(images.shape) # (1024, 1024)
    return ids, images, labels
def label_dict(labels:pd.Series):
    label_dict = {}
    for i, label in enumerate(labels.unique()):
        label_dict[label] = i
    return label_dict
def label_encoding(labels:pd.Series, label_dict:dict):
    return labels.map(label_dict)
def save_image(ids, images, labels, path):
    # 이미지는 (32, 32)로 저장
    path = os.path.join(os.getcwd(), path)
    for i in range(len(ids)):
        img = images[i]
        label = labels[i]
        img = img.astype(np.uint8)
        cv2.imwrite(os.path.join(path, f'{ids[i]}_{label}.png'), img)
    print('이미지 저장 완료')

# 데이터 불러오기
if not os.getcwd().endswith('data'):
    os.chdir('./image_icon_25_03/data')
df = pd.read_csv('train.csv')
ids, images, labels = split_reshape(df)
label_encode = label_dict(labels)
labels = label_encoding(labels, label_encode)
json.dump(label_encode, open('label_encode.json', 'w')) # label_encode.json 파일로 저장
save_image(ids, images, labels, 'train')
df = pd.read_csv('test.csv')
ids, images, labels = split_reshape(df)
labels = np.zeros(len(ids), dtype=np.int64) # test 데이터는 label이 없으므로 0으로 채움
save_image(ids, images, labels, 'test')
