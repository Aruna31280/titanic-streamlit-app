import streamlit as st
import pandas as pd
import pickle

# Load datasets
raw_df = pd.read_csv("Titanic_train.csv")

# For cleaned dataset, you can either save it from notebook or preprocess here
cleaned_df = raw_df.copy()
# Minimal preprocessing example:
cleaned_df["Sex"] = cleaned_df["Sex"].map({"male":0, "female":1})
cleaned_df = cleaned_df.dropna(subset=["Age","Fare","Pclass","SibSp","Parch"])

# Load trained model
model = pickle.load(open("logistic_regression_assignment.pkl", "rb"))

# Title
st.title("Titanic Dataset Viewer")

# Display Raw Dataset
st.subheader("Raw Dataset")
st.dataframe(raw_df)

# Display Cleaned Dataset
st.subheader("Cleaned Dataset")
st.dataframe(Cleaned_df)

# Prediction Section
st.subheader("Predict Passenger Survival")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sex = st.selectbox("Sex", ["male", "female"])
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)

sex_encoded = 1 if sex == "female" else 0

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "Fare": [fare],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Sex": [sex_encoded]
})

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Passenger is LIKELY TO SURVIVE")
    else:
        st.error("Passenger is UNLIKELY TO SURVIVE")
