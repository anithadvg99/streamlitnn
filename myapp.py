import streamlit as st
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = './networkmodel.h5'
model = tf.keras.models.load_model(model_path)

# Streamlit App
st.title("Predicting data for numerical Data")

# Input for numerical data
num_input = st.text_input("Enter numerical data (comma-separated):")

# Convert input string to a numerical array


if st.button("Predict"):
    
        # Make predictions
        tdata=np.array([[85,58,41,21.77046169,80.31964408,7.038096361,226.6555374]])
        predictions = model.predict(tdata)

        # Display the predictions
        st.subheader("Predictions:")
        for i, pred in enumerate(predictions[0]):
            st.write(f"Output{i + 1}: {pred:.4f}")

        # Display the most likely output
        predicted_output = np.argmax(predictions[0]) + 1
        st.subheader(f"Predicted Output: {predicted_output}")