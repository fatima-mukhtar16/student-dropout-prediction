import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

st.title("🎓 Student Dropout Risk Predictor")
st.write("Enter student details to predict dropout risk.")

@st.cache_data
def load_and_train():
    dataset = fetch_ucirepo(id=697)
    X = dataset.data.features
    y_raw = dataset.data.targets
    df = pd.concat([X, y_raw], axis=1)
    df['Dropout_binary'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)
    
    X_final = df.drop(columns=['Target', 'Dropout_binary'])
    y_final = df['Dropout_binary']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y_final)
    
    return model, scaler, X_final.columns.tolist()

model, scaler, feature_names = load_and_train()

st.subheader("Enter Student Information")

admission_grade = st.slider("Admission Grade", 0, 200, 120)
sem1_grade = st.slider("1st Semester Grade", 0, 20, 12)
sem2_grade = st.slider("2nd Semester Grade", 0, 20, 12)
tuition_paid = st.selectbox("Tuition Fees Up to Date?", [1, 0])
scholarship = st.selectbox("Scholarship Holder?", [0, 1])
age = st.slider("Age at Enrollment", 17, 50, 20)

if st.button("Predict Dropout Risk"):
    sample = {col: 0 for col in feature_names}
    sample['Admission grade'] = admission_grade
    sample['Curricular units 1st sem (grade)'] = sem1_grade
    sample['Curricular units 2nd sem (grade)'] = sem2_grade
    sample['Tuition fees up to date'] = tuition_paid
    sample['Scholarship holder'] = scholarship
    sample['Age at enrollment'] = age
    sample['Curricular units 1st sem (enrolled)'] = 6
    sample['Curricular units 2nd sem (enrolled)'] = 6
    sample['Curricular units 1st sem (approved)'] = 5
    sample['Curricular units 2nd sem (approved)'] = 5
    
    input_df = pd.DataFrame([sample], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]
    
    if probability < 0.3:
        risk = "🟢 Low Risk"
    elif probability < 0.6:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"
    
    st.subheader("Result")
    st.write(f"**Predicted Class:** {'Dropout' if prediction == 1 else 'Not Dropout'}")
    st.write(f"**Dropout Probability:** {round(probability*100, 2)}%")
    st.write(f"**Risk Level:** {risk}")