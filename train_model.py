# train_model.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# 입력 데이터 (X): [shape_code, mass, asymmetry]
X = np.array([
    [0, 1.0, 0.0],  # cube, 대칭
    [0, 1.0, 0.9],  # cube, 매우 비대칭
    [1, 1.0, 0.0],  # cylinder, 대칭
    [1, 1.0, 0.6],  # cylinder, 약간 기울어짐
    [0, 2.0, 0.7],  # cube, 무거움
    [1, 0.5, 0.8],  # cylinder, 가볍고 비대칭
])

# 출력 데이터 (y): [0=세워짐, 1=옆으로 누움]
y = [0, 1, 0, 1, 1, 1]

# 모델 학습
model = RandomForestClassifier()
model.fit(X, y)

# 저장
joblib.dump(model, "orientation_model.pkl")
print("✅ orientation_model.pkl 저장 완료!")
