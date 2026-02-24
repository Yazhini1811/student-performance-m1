import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Student Performance Prediction")

# Dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [35,40,50,55,65,70,75,85,90,95]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

hours = st.slider("Select study hours", 0, 12, 5)

prediction = model.predict([[hours]])

st.write(f"Predicted Marks: {prediction[0]:.2f}")

# Graph
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, model.predict(X))
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")

st.pyplot(fig)