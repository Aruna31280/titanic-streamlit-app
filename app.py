import streamlit as st
import pandas as pd
import pickle

# Load trained logistic regression assignment
model = pickle.load(open("logistic_regression_assignment.pkl", "rb"))

# Streamlit app
st.title("Titanic Survival Prediction (Logistic Regression)")

# User inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Passenger Fare", 0.0, 600.0, 32.0)

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Passenger is LIKELY TO SURVIVE")
    else:
        st.error("Passenger is UNLIKELY TO SURVIVE")
