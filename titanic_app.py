import streamlit as st
import joblib
import numpy as np


model = joblib.load('titanic_model.pkl')


st.title("Titanic Survival Prediction 🚢")


pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.slider("Parents/Children Aboard", 0, 5, 0)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["Cherbourg", "Queenstown", "Southamptown"])


sex_encoded = 1 if sex == 'male' else 0
embarked_encoded = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}[embarked]


features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])


if st.button("Predict Survival"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("🎉 The passenger is predicted to SURVIVE!")
    else:
        st.error("😢 The passenger is predicted to NOT survive.")