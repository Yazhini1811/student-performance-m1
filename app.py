import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("🎓 Student Performance Prediction")

st.write("Predict marks based on study hours using Machine Learning.")

# Dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [35,40,50,55,65,70,75,85,90,95]
}

df = pd.DataFrame(data)

# Model Training
X = df[["Hours_Studied"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

# User Input
hours = st.slider("Select Study Hours", 0, 12, 5)

# Prediction
prediction = model.predict([[hours]])

st.subheader("Predicted Marks")
st.success(f"{prediction[0]:.2f}")

# Visualization
st.subheader("Study Hours vs Marks")

fig, ax = plt.subplots()

ax.scatter(X, y)
ax.plot(X, model.predict(X))

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
ax.set_title("Student Performance Prediction")

st.pyplot(fig)

# Show Dataset
st.subheader("Dataset")
st.dataframe(df)
