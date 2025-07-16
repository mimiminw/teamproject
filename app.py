import streamlit as st
import pybullet as p
import pybullet_data
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="3D 도형 낙하 시뮬레이션", layout="centered")
st.title("🪂 3D 도형 낙하 시뮬레이션 (ML 제외)")

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

    for _ in range(240):  # 약 4초 시뮬레이션
        p.stepSimulation()

    pos, orn = p.getBasePositionAndOrientation(body)
    p.disconnect()
    return pos, orn

def plot_shape(shape='cube'):
    fig = go.Figure()

    if shape == "cube":
        # 정육면체 꼭짓점
        x = [0,1,1,0,0,1,1,0]
        y = [0,0,1,1,0,0,1,1]
        z = [0,0,0,0,1,1,1,1]
        # Mesh3d는 삼각형 면을 i,j,k로 지정 (0-based vertex 인덱스)
        i = [0,0,0,1,1,2,4,5,6,7,7,3]
        j = [1,2,3,5,6,3,5,6,7,4,3,2]
        k = [2,3,1,6,7,0,6,7,4,0,0,1]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
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

shape = st.selectbox("도형 선택", ["cube", "cylinder"])
mass = st.slider("질량", 0.1, 10.0, 1.0)

if st.button("💥 물리 시뮬레이션 실행"):
    pos, orn = run_simulation(shape, mass)
    if pos and orn:
        st.success(f"도형 위치: {pos}")
        st.success(f"자세 (Quaternion): {orn}")
        fig = plot_shape(shape)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("지원하지 않는 도형입니다.")
