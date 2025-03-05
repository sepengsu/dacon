import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# 모델 정의 (저장된 모델과 동일한 구조 사용)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x  # Softmax를 사용하지 않음 (CrossEntropyLoss가 알아서 처리)

# 모델 로드 함수
def load_model(model_path, device):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드로 설정 (Dropout, BatchNorm 해제)
    return model

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 흑백 로드
    img = cv2.resize(img, (32, 32))  # 모델 입력 크기로 변경
    img = np.expand_dims(img, axis=0)  # 채널 차원 추가 (1, 32, 32)
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가 (1, 1, 32, 32)
    img = img.astype(np.float32) / 255.0  # 정규화
    return torch.tensor(img, dtype=torch.float32)

# 클래스 예측 함수
def predict(image_path, model, device):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():  # No gradient computation
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()  # 가장 높은 확률의 클래스 선택
    return predicted_class

# 실행 예제
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "saved_models/best.pth"  # 저장된 모델 경로
    model = load_model(model_path, device)
    
    test_image = "test_image.png"  # 예측할 이미지 경로
    predicted_class = predict(test_image, model, device)
    
    print(f"Predicted Class: {predicted_class}")
