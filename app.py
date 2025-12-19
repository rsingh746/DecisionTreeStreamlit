import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Decision Trees: Geometry and Failure Modes")

# -------------------------------------------------------------------
# Cached data utilities
# -------------------------------------------------------------------

@st.cache_data
def generate_staircase_data(n_points=300):
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y_true = np.sin(X).ravel()
    return X, y_true


@st.cache_data
def generate_xor_data(n=300, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)

    if noise > 0:
        flip_mask = rng.rand(n) < noise
        y[flip_mask] = 1 - y[flip_mask]

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


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

experiment = st.sidebar.selectbox(
    "Experiment",
    ["Staircase Effect", "XOR Blindspot"]
)

max_depth = st.sidebar.slider("Tree depth", 1, 10, 3)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.0, step=0.05)

# -------------------------------------------------------------------
# Staircase Effect (Regression)
# -------------------------------------------------------------------

if experiment == "Staircase Effect":

    st.header("Staircase Effect in Regression Trees")

    X, y_true = generate_staircase_data()
    rng = np.random.RandomState(0)
    y_obs = y_true + noise * rng.randn(len(y_true))

    model = train_regression_tree(X, y_obs, max_depth)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        X,
        y_true,
        linewidth=2,
        label="True function",
        zorder=3
    )

    ax.scatter(
        X,
        y_obs,
        s=12,
        alpha=0.4,
        label="Noisy observations",
        zorder=2
    )

    ax.step(
        X.ravel(),
        y_pred,
        where="post",
        linewidth=2,
        label="Tree prediction",
        zorder=4
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Piecewise Constant Approximation")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig, clear_figure=True)

    st.markdown(
        "Noise does not change the underlying function. "
        "It changes the observations the model sees. "
        "As depth increases, the tree becomes capable of fitting these fluctuations."
    )

# -------------------------------------------------------------------
# XOR Blindspot (Classification)
# -------------------------------------------------------------------

else:

    st.header("XOR Blindspot in Classification Trees")

    X, y = generate_xor_data(noise=noise)
    xx, yy, grid = generate_mesh()

    model = train_classification_tree(X, y, max_depth)
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.contourf(xx, yy, Z, alpha=0.3)

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        alpha=0.8,
        label="Class 0",
        edgecolor="k"
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        alpha=0.8,
        label="Class 1",
        edgecolor="k"
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Decision Boundary (depth = {max_depth})")
    ax.legend()

    st.pyplot(fig, clear_figure=True)

    if noise > 0:
        st.markdown(
            "Label noise introduces contradictions in the data. "
            "With sufficient depth, the tree responds by carving small regions "
            "around mislabeled points."
        )
    else:
        st.markdown(
            "With clean labels, increasing depth allows the tree to recover "
            "the XOR interaction by stacking greedy splits."
        )
