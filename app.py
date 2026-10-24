import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "student_grade_model.joblib"

st.set_page_config(page_title="Grade Forecaster", page_icon="📊", layout="wide")


@st.cache_resource
def load_model():
    # If the model file is missing (or made with another scikit-learn version),
    # train it fresh from the dataset.
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, "train_model.py"], check=True)
        return joblib.load(MODEL_PATH)


model = load_model()

st.title("📊 Grade Forecaster")
st.markdown("Estimate a student's **final maths grade (G3, out of 20)** from their profile.")

yn = ["yes", "no"]

with st.sidebar:
    st.header("About")
    st.write(
        "Model trained on the UCI *Student Performance* dataset (395 maths students, Portugal). "
        "Predictions are estimates for learning purposes only."
    )

tab_acad, tab_home, tab_life = st.tabs(["🎓 Academic", "🏠 Home & Family", "🎯 Lifestyle"])

with tab_acad:
    c1, c2, c3 = st.columns(3)
    G1 = c1.slider("Period 1 grade (G1)", 0, 20, 10)
    G2 = c2.slider("Period 2 grade (G2)", 0, 20, 10)
    absences = c3.number_input("Absences", 0, 100, 4)
    c1, c2, c3 = st.columns(3)
    studytime = c1.select_slider("Weekly study time", [1, 2, 3, 4],
                                 help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
    failures = c2.select_slider("Past failures", [0, 1, 2, 3])
    higher = c3.radio("Wants higher education", yn, horizontal=True)
    c1, c2, c3 = st.columns(3)
    school = c1.selectbox("School", ["GP", "MS"])
    schoolsup = c2.radio("School extra support", yn, index=1, horizontal=True)
    paid = c3.radio("Paid extra classes", yn, index=1, horizontal=True)

with tab_home:
    c1, c2, c3 = st.columns(3)
    sex = c1.selectbox("Sex", ["F", "M"])
    age = c2.number_input("Age", 15, 22, 17)
    address = c3.selectbox("Address", ["U", "R"], format_func=lambda x: {"U": "Urban", "R": "Rural"}[x])
    c1, c2, c3 = st.columns(3)
    famsize = c1.selectbox("Family size", ["GT3", "LE3"], format_func=lambda x: {"GT3": "> 3", "LE3": "≤ 3"}[x])
    Pstatus = c2.selectbox("Parents", ["T", "A"], format_func=lambda x: {"T": "Together", "A": "Apart"}[x])
    guardian = c3.selectbox("Guardian", ["mother", "father", "other"])
    c1, c2 = st.columns(2)
    Medu = c1.select_slider("Mother's education", [0, 1, 2, 3, 4], value=2)
    Fedu = c2.select_slider("Father's education", [0, 1, 2, 3, 4], value=2)
    jobs = ["teacher", "health", "services", "at_home", "other"]
    c1, c2, c3 = st.columns(3)
    Mjob = c1.selectbox("Mother's job", jobs, index=4)
    Fjob = c2.selectbox("Father's job", jobs, index=4)
    reason = c3.selectbox("Reason for school", ["course", "home", "reputation", "other"])
    c1, c2, c3 = st.columns(3)
    famsup = c1.radio("Family support", yn, horizontal=True)
    internet = c2.radio("Internet at home", yn, horizontal=True)
    traveltime = c3.select_slider("Travel time", [1, 2, 3, 4], help="1: <15min ... 4: >1h")

with tab_life:
    c1, c2, c3 = st.columns(3)
    famrel = c1.slider("Family relationship", 1, 5, 4)
    freetime = c2.slider("Free time", 1, 5, 3)
    goout = c3.slider("Going out", 1, 5, 3)
    c1, c2, c3 = st.columns(3)
    Dalc = c1.slider("Workday alcohol", 1, 5, 1)
    Walc = c2.slider("Weekend alcohol", 1, 5, 2)
    health = c3.slider("Health", 1, 5, 3)
    c1, c2, c3 = st.columns(3)
    activities = c1.radio("Extracurriculars", yn, horizontal=True)
    nursery = c2.radio("Attended nursery", yn, horizontal=True)
    romantic = c3.radio("In a relationship", yn, index=1, horizontal=True)

if st.button("🔮 Predict final grade", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
        "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
        "reason": reason, "guardian": guardian, "traveltime": traveltime,
        "studytime": studytime, "failures": failures, "schoolsup": schoolsup,
        "famsup": famsup, "paid": paid, "activities": activities, "nursery": nursery,
        "higher": higher, "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences, "G1": G1, "G2": G2,
    }])
    grade = float(min(20, max(0, model.predict(row)[0])))

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted G3", f"{grade:.1f} / 20")
    m2.metric("Percentage", f"{grade / 20 * 100:.0f}%")
    m3.metric("Change vs G2", f"{grade - G2:+.1f}")
    st.progress(grade / 20)

    if grade >= 15:
        st.success("Excellent: on track for a strong result.")
    elif grade >= 10:
        st.info("Passing: there is room to improve.")
    else:
        st.warning("At risk of failing: extra support may help.")
