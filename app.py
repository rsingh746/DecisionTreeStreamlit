import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Decision Trees: Geometry and Failure Modes")


@st.cache_data
def generate_staircase_data(n_points=300):
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y_true = np.sin(X).ravel()
    return X, y_true


@st.cache_data
def generate_xor_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    return X, y


@st.cache_data
def generate_mesh():
    xx, yy = np.meshgrid(
        np.linspace(0, 1, 250),
        np.linspace(0, 1, 250)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


@st.cache_resource
def train_regression_tree(X, y, depth):
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(X, y)
    return model


@st.cache_resource
def train_classification_tree(X, y, depth):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X, y)
    return model


experiment = st.sidebar.selectbox(
    "Experiment",
    ["Staircase Effect", "XOR Blindspot"]
)

max_depth = st.sidebar.slider("Tree depth", 1, 10, 3)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.0, step=0.05)

if experiment == "Staircase Effect":

    st.header("Staircase Effect in Regression Trees")

    X, y_true = generate_staircase_data()
    y = y_true + noise * np.random.randn(len(y_true))

    model = train_regression_tree(X, y, max_depth)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(X, y_true, linewidth=2, label="True function")
    ax.step(X.ravel(), y_pred, where="post", linewidth=2, label="Tree prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Piecewise Constant Approximation")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig, clear_figure=True)

    st.markdown(
        "Increasing depth creates finer steps, but the prediction never becomes smooth. "
        "Decision trees approximate functions using flat regions separated by jumps."
    )

else:

    st.header("XOR Blindspot in Classification Trees")

    X, y = generate_xor_data()
    xx, yy, grid = generate_mesh()

    model = train_classification_tree(X, y, max_depth)
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", alpha=0.8)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Decision boundary (depth = {max_depth})")

    st.pyplot(fig, clear_figure=True)

    if max_depth == 1:
        st.markdown(
            "At depth 1, no split reduces impurity. "
            "Each feature looks uninformative on its own."
        )
    else:
        st.markdown(
            "With sufficient depth, the tree combines splits and recovers the interaction."
        )
