import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pybullet as p
import pybullet_data
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(page_title="3D 도형 낙하 예측기", layout="centered")
st.title("🪂 3D 도형 낙하 예측기")

# -------------------------
# ML 모델 초기화 및 학습
# -------------------------
MODEL_PATH = "orientation_model.pkl"

def train_dummy_model():
    X = np.array([
        [0, 1.0, 0.0],
        [0, 1.0, 0.8],
        [1, 1.0, 0.0],
        [1, 1.0, 0.6]
    ])
    y = [0, 1, 0, 1]  # 0=세워짐, 1=옆으로 누움
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    train_dummy_model()

def predict_orientation(shape, mass, asymmetry):
    model = joblib.load(MODEL_PATH)
    shape_map = {"cube": 0, "cylinder": 1}
    x = np.array([[shape_map[shape], mass, asymmetry]])
    return model.predict(x)[0]

# -------------------------
# PyBullet 시뮬레이션
# -------------------------
def run_simulation(shape='cube', mass=1.0):
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    if shape == 'cube':
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    elif shape == 'cylinder':
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.0)
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.0)
    else:
        return None, None

    body = p.createMultiBody(baseMass=mass,
                             baseCollisionShapeIndex=collision,
                             baseVisualShapeIndex=visual,
                             basePosition=[0, 0, 5])

    for _ in range(240):  # 4초간 시뮬레이션
        p.stepSimulation()

    pos, orn = p.getBasePositionAndOrientation(body)
    p.disconnect()
    return pos, orn

# -------------------------
# Plotly 3D 시각화
# -------------------------
def plot_shape(shape='cube', orientation=[0,0,0,1]):
    fig = go.Figure()

    if shape == "cube":
        fig.add_trace(go.Mesh3d(
            x=[0,1,1,0,0,1,1,0],
            y=[0,0,1,1,0,0,1,1],
            z=[0,0,0,0,1,1,1,1],
            i=[0,0,0,1,1,2,3,4,4,5,6,7],
            j=[1,2,4,3,5,3,0,5,6,6,7,6],
            k=[2,4,5,2,6,0,1,6,7,7,4,5],
            color='skyblue',
            opacity=0.5
        ))
    elif shape == "cylinder":
        theta = np.linspace(0, 2 * np.pi, 30)
        x = 0.5 * np.cos(theta)
        y = 0.5 * np.sin(theta)
        z = np.zeros_like(theta)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='orange')))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z+1, mode='lines', line=dict(color='orange')))

    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[-2, 2]),
        yaxis=dict(nticks=4, range=[-2, 2]),
        zaxis=dict(nticks=4, range=[-1, 5]),
    ))
    return fig

# -------------------------
# Streamlit UI
# -------------------------
shape = st.selectbox("도형 선택", ["cube", "cylinder"])
mass = st.slider("질량", 0.1, 10.0, 1.0)
asymmetry = st.slider("무게중심 비대칭 정도", 0.0, 1.0, 0.0)

col1, col2 = st.columns(2)

with col1:
    if st.button("💥 물리 시뮬레이션 실행"):
        pos, orn = run_simulation(shape, mass)
        st.success(f"도형 위치: {pos}")
        st.success(f"자세 (Quaternion): {orn}")
        fig = plot_shape(shape, orn)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if st.button("🧠 ML 예측 실행"):
        pred = predict_orientation(shape, mass, asymmetry)
        label = "세워짐" if pred == 0 else "옆으로 누움"
        st.success(f"ML 예측 결과: **{label}**")
