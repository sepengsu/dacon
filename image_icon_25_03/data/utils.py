import os , cv2

def get_imageandlabel(file_name, data_dir):
    '''
    input: 이미지 경로 중 마지막 파일명 
    이름명: TRAIN_번호_라벨.png
    output: 이미지 경로, 라벨
    '''
    path = os.path.join(data_dir, file_name) # 이미지 경로
    name = file_name.replace('.png', '') # png 확장자 제거
    class_name = int(name.split('_')[-1]) # 라벨
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 이미지 불러오기
    return image, class_name

