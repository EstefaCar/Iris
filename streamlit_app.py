
import streamlit as st
import joblib
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Species Predictor 🌸",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="auto", # Can be "auto", "expanded", "collapsed"
    menu_items={
        'Get Help': 'https://www.streamlit.io/help',
        'Report a bug': "https://www.github.com/streamlit/streamlit/issues",
        'About': "# This is a simple Iris species prediction app built with Streamlit and a KNN model."
    }
)

# --- Title and Introduction ---
st.markdown("<h1 style='text-align: center; color: #4B4B4B;'>🌸 Iris Flower Species Predictor 🌸</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #696969;'>Uncover the species of Iris flowers by providing their measurements. This app uses a pre-trained K-Nearest Neighbors (KNN) model.</p>", unsafe_allow_html=True)
st.write("---")

# --- Model Loading ---
@st.cache_resource # Cache the model loading for better performance
def load_model():
    try:
        model_path = 'best_knn_model.pkl'
        model = joblib.load(model_path)
        # st.success("Model loaded successfully!") # Removed as it's less critical for the user to see this message
        return model
    except Exception as e:
        st.error(f"⚠️ Error: Could not load the prediction model. Please ensure 'best_knn_model.pkl' is in the correct directory. Error details: {e}")
        st.stop() # Stop the app if model can't be loaded

loaded_model = load_model()

# Define species names (assuming typical Iris dataset labels 0:Setosa, 1:Versicolor, 2:Virginica)
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

st.markdown("<h3>Input Flower Measurements:</h3>", unsafe_allow_html=True)

# --- Input Widgets for Features with Columns ---
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("📏 Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.slider("↔️ Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)

with col2:
    petal_length = st.slider("🌿 Petal Length (cm)", 1.0, 7.0, 1.3, 0.1)
    petal_width = st.slider("💧 Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

# Create a feature array for prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

st.write("---")

# --- Prediction Button and Output ---
if st.button("✨ Predict Species", help="Click to predict the Iris species based on the provided measurements"):
    with st.spinner("Predicting..."):
        prediction_numeric = loaded_model.predict(features)[0]
        predicted_species = species_map.get(prediction_numeric, "Unknown")

    st.markdown(f"<h2 style='text-align: center; color: #3CB371;'>🎉 Predicted Species: <span style='font-weight: bold;'>{predicted_species}</span></h2>", unsafe_allow_html=True)
    if predicted_species == 'Setosa':
        st.image("https://en.wikipedia.org/wiki/Iris_setosa#/media/File:Irissetosa1.jpg", caption="Iris Setosa", use_column_width=True)
    elif predicted_species == 'Versicolor':
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor", use_column_width=True)
    elif predicted_species == 'Virginica':
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica", use_column_width=True)
    else:
        st.info("No image available for unknown species.")

st.markdown("---")

# --- Footer ---
st.info("💡 This application uses a K-Nearest Neighbors (KNN) model trained on the classic Iris dataset for classification.")
st.markdown("<p style='text-align: center; color: #A9A9A9; font-size: small;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
