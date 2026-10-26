import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Student Risk Evaluation System", layout="wide", page_icon="🎓")

# Load Serialized Assets
model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Ensemble ML Framework for Student Performance Prediction")
st.caption("Proactive Academic Intervention Tool | Aligned with SDG 4 (Quality Education)")

tab1, tab2 = st.tabs(["👤 Single Student Assessment", "📂 Class Cohort Batch Evaluation"])

with tab1:
    st.subheader("Enter Intra-Semester Student Metrics")
    c1, c2 = st.columns(2)
    
    with c1:
        t1 = st.slider("Continuous Assessment 1 (out of 15)", 0.0, 15.0, 8.0)
        t2 = st.slider("Continuous Assessment 2 (out of 15)", 0.0, 15.0, 7.5)
        assign = st.slider("Assignment/Lab Score (out of 10)", 0.0, 10.0, 6.0)
        
    with c2:
        att = st.slider("Lecture Attendance Rate (%)", 0.0, 100.0, 65.0)
        logins = st.number_input("Weekly LMS Logins", min_value=0, max_value=100, value=8)
        posts = st.number_input("LMS Forum Contributions", min_value=0, max_value=50, value=1)
        
    if st.button("Run Risk Prediction"):
        input_data = np.array([[t1, t2, assign, att, logins, posts]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probs = model.predict_proba(scaled_data)[0]
        
        st.markdown("---")
        if prediction == 0:
            st.error(f"⚠️ **STATUS: HIGH ACADEMIC RISK DETECTED** (Model Confidence: {probs[0]*100:.2f}%)")
            st.info("📌 **Recommended Action:** Refer student to Departmental Academic Advisor and assign peer tutoring.")
        else:
            st.success(f"✅ **STATUS: STUDENT IS ON TRACK** (Model Confidence: {probs[1]*100:.2f}%)")

with tab2:
    st.subheader("Upload Departmental Roster (CSV)")
    uploaded_file = st.file_uploader("Upload CSV containing columns: test1, test2, assignment, attendance, lms_logins, forum_posts", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Roster Preview:", batch_df.head())
        
        feature_cols = ['test1', 'test2', 'assignment', 'attendance', 'lms_logins', 'forum_posts']
        scaled_batch = scaler.transform(batch_df[feature_cols])
        batch_preds = model.predict(scaled_batch)
        
        batch_df['Risk_Category'] = np.where(batch_preds == 0, 'High Risk ⚠️', 'On Track ✅')
        
        st.subheader("Prediction Summary")
        st.dataframe(batch_df)
        
        csv_export = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Evaluated Roster CSV", csv_export, "evaluated_students.csv", "text/csv")
