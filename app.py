import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Decision Trees: Geometry and Failure Modes")

experiment = st.sidebar.selectbox(
    "Experiment",
    ["Staircase Effect", "XOR Blindspot"]
)

max_depth = st.sidebar.slider("Tree depth", 1, 10, 3)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.0, step=0.05)

if experiment == "Staircase Effect":

    st.header("Staircase Effect in Regression Trees")

    X = np.linspace(0, 10, 300).reshape(-1, 1)
    y_true = np.sin(X).ravel()
    y = y_true + noise * np.random.randn(len(y_true))

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(X, y_true, linewidth=2, label="True function")
    ax.step(X.ravel(), y_pred, where="post", linewidth=2, label="Tree prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Piecewise Constant Approximation")
    ax.legend()
    ax.grid(True)

    st.markdown(
        "Increasing depth creates finer steps, but the prediction never becomes smooth. "
        "Decision trees approximate functions using flat regions separated by jumps."
    )

    st.pyplot(fig)

else:

    st.header("XOR Blindspot in Classification Trees")

    np.random.seed(0)
    n = 300
    X = np.random.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)

    xx, yy = np.meshgrid(
        np.linspace(0, 1, 300),
        np.linspace(0, 1, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", alpha=0.8)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Decision boundary (depth = {max_depth})")

    if max_depth == 1:
        st.markdown(
            "At depth 1, no split reduces impurity. "
            "Each feature looks uninformative on its own."
        )
    else:
        st.markdown(
            "With sufficient depth, the tree combines splits and recovers the interaction."
        )
    st.pyplot(fig)
