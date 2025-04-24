import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time 
st.set_page_config(page_title="Student Score Predictor", layout="wide", initial_sidebar_state="collapsed")


page_bg_color = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-color: #90EE90; /* Standard LightGreen */
}
/* body { background-color: #90EE90; } */ /* Alternative selector */
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True) 



MODEL_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_student_performance_v1_model.pkl')
CAT_INDICES_PATH = os.path.join(MODEL_DIR, 'catboost_categorical_feature_indices_v1.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'catboost_final_training_columns_v1.pkl')


SCALING_RANGES = {
    'sleep_hours': (4, 12),
    'exercise_frequency': (0, 7),
    'mental_health_rating': (1, 5)
}


@st.cache_resource
def load_artifacts():
   
    
    if not os.path.exists(MODEL_DIR):
         st.error(f"ERROR: Directory '{MODEL_DIR}' not found. Please ensure it exists in the same folder as app.py.")
         return None, None, None
    if not os.path.exists(MODEL_PATH):
        st.error(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        return None, None, None
    if not os.path.exists(CAT_INDICES_PATH):
         st.error(f"ERROR: Categorical indices file not found at '{CAT_INDICES_PATH}'.")
         return None, None, None
    if not os.path.exists(COLUMNS_PATH):
         st.error(f"ERROR: Columns file not found at '{COLUMNS_PATH}'.")
         return None, None, None

    try:
        model = joblib.load(MODEL_PATH)
        cat_indices = joblib.load(CAT_INDICES_PATH)
        columns = joblib.load(COLUMNS_PATH)
        print("Artifacts loaded successfully.") 
        return model, cat_indices, columns
    except Exception as e:
        st.error(f"Error loading artifacts from '{MODEL_DIR}'. Error: {e}")
        return None, None, None

model, cat_indices, final_columns = load_artifacts()

if model is None or cat_indices is None or final_columns is None:
    st.warning("Model artifacts could not be loaded or found. App cannot proceed.")
    st.stop()
def preprocess_and_engineer(input_data_dict):
   
    try:
        df = pd.DataFrame([input_data_dict])

        # 1. Binary Encoding
        binary_cols = ['part_time_job', 'extracurricular_participation']
        for col in binary_cols:
            if col in df.columns:
                 df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

        # 2. Handle Categorical Values
        cat_feature_names = [final_columns[i] for i in cat_indices if i < len(final_columns)]
        for col in cat_feature_names:
             if col in df.columns:
                  if df[col].iloc[0] is None or df[col].iloc[0] == '': df[col] = "Missing"
                  else: df[col] = df[col].astype(str)
             else: df[col] = "Missing"

        # 3. Feature Engineering
        cols_to_drop_from_fe = []
        new_features_created = []

        # Scaled Wellbeing Index
        wellbeing_components = ['sleep_hours', 'exercise_frequency', 'mental_health_rating']
        if all(col in df.columns for col in wellbeing_components):
            scaled_wellbeing_sum = 0
            for col in wellbeing_components:
                min_val, max_val = SCALING_RANGES[col]
                value = df[col].iloc[0]
                if pd.isna(value): value = (min_val + max_val) / 2
                scaled_val = np.clip((value - min_val) / (max_val - min_val + 1e-6), 0, 1)
                scaled_wellbeing_sum += scaled_val
            df['wellbeing_index'] = scaled_wellbeing_sum
            new_features_created.append('wellbeing_index')
            cols_to_drop_from_fe.extend(wellbeing_components)

        # Study Consistency
        consistency_components = ['study_hours_per_day', 'attendance_percentage']
        if all(col in df.columns for col in consistency_components):
            study_hours = pd.to_numeric(df['study_hours_per_day'].iloc[0], errors='coerce')
            attendance = pd.to_numeric(df['attendance_percentage'].iloc[0], errors='coerce')
            if pd.notna(study_hours) and pd.notna(attendance):
                 df['study_consistency'] = study_hours * (attendance / 100.0)
            else: df['study_consistency'] = 0
            new_features_created.append('study_consistency')
            cols_to_drop_from_fe.extend(consistency_components)

        # Total Screen Time
        screen_time_components = ['netflix_hours', 'social_media_hours']
        if all(col in df.columns for col in screen_time_components):
             netflix_h = pd.to_numeric(df['netflix_hours'].iloc[0], errors='coerce')
             social_h = pd.to_numeric(df['social_media_hours'].iloc[0], errors='coerce')
             df['total_screen_time'] = (netflix_h if pd.notna(netflix_h) else 0) + \
                                       (social_h if pd.notna(social_h) else 0)
             new_features_created.append('total_screen_time')
             cols_to_drop_from_fe.extend(screen_time_components)

        # 4. Drop original numerical columns 
        unique_cols_to_drop = list(set(cols_to_drop_from_fe))
        existing_cols_to_drop = [col for col in unique_cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df.drop(columns=existing_cols_to_drop, inplace=True)

        # 5. Ensure Final Columns Match Training Columns EXACTLY
        for col in final_columns:
            if col not in df.columns:
                if col in cat_feature_names: df[col] = "Missing"
                else: df[col] = 0
        df = df[final_columns] # Reorder

        # 6. Handle remaining numerical NaNs
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if df[col].isnull().any(): df[col].fillna(0, inplace=True)

        return df

    except Exception as e:
        st.error(f"Error during preprocessing/feature engineering: {e}")
        st.json(input_data_dict)
        return None



st.title("🎓 Student Exam Score Predictor")
st.markdown("Enter student details to predict their potential exam score based on habits and background.")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Student Information & Habits")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Background**")
        age = st.number_input("Age", min_value=17, max_value=24, value=20, step=1, help="Student's age (17-24).")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0, help="Student's gender.")
        parental_education_level = st.selectbox("Parental Education Level",
                                                ["None", "High School", "Some College", "Bachelor's Degree", "Master's Degree", "PhD"],
                                                index=1, help="Highest education level of parents.")
        part_time_job = st.radio("Has Part-Time Job?", ["No", "Yes"], index=0, horizontal=True)

  
    with col2:
        st.markdown("**Academic Habits**")
        study_hours_per_day_in = st.slider("Study Hours per Day", min_value=0.0, max_value=12.0, value=3.0, step=0.5)
        attendance_percentage_in = st.slider("Attendance Percentage (%)", min_value=50, max_value=100, value=85, step=1, help="Attendance (50-100%).")
        internet_quality = st.selectbox("Internet Quality at Home", ["Poor", "Average", "Good"], index=1)
        extracurricular_participation = st.radio("Participates in Extracurriculars?", ["No", "Yes"], index=1, horizontal=True)

    with col3:
        st.markdown("**Lifestyle & Wellbeing**")
        sleep_hours_in = st.slider("Avg Sleep Hours per Night", min_value=4.0, max_value=12.0, value=7.0, step=0.5, help="Sleep (4-12 hours).")
        social_media_hours_in = st.slider("Hours/Day on Social Media", min_value=0.0, max_value=8.0, value=2.0, step=0.5, help="Social Media (0-8 hours).")
        exercise_frequency_in = st.slider("Exercise Frequency (Times/Week)", min_value=0, max_value=7, value=3, step=1)
        netflix_hours_in = st.slider("Hours/Day Streaming (Netflix etc.)", min_value=0.0, max_value=12.0, value=1.5, step=0.5)
        mental_health_rating_in = st.slider("Mental Health Rating (1-5)", min_value=1, max_value=5, value=3, step=1, help="1=Low, 5=High")
        diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"], index=1)

    submitted = st.form_submit_button("Predict Exam Score")

if submitted:
    with st.spinner("Analysing habits and predicting score..."):
        input_data = {
            'age': age, 'gender': gender, 'parental_education_level': parental_education_level,
            'part_time_job': part_time_job, 'study_hours_per_day': study_hours_per_day_in,
            'attendance_percentage': attendance_percentage_in, 'internet_quality': internet_quality,
            'extracurricular_participation': extracurricular_participation, 'sleep_hours': sleep_hours_in,
            'social_media_hours': social_media_hours_in, 'exercise_frequency': exercise_frequency_in,
            'netflix_hours': netflix_hours_in, 'mental_health_rating': mental_health_rating_in,
            'diet_quality': diet_quality
        }

        processed_df = preprocess_and_engineer(input_data)

        if processed_df is not None:
            try:
                prediction = model.predict(processed_df)
                raw_prediction = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)
                clipped_score = np.clip(raw_prediction, 15.0, 100.0)
                display_score = round(clipped_score, 1)

                st.markdown("---")
                st.subheader("Prediction Result")
                st.metric(label="Predicted Exam Score", value=f"{display_score:.1f}")

                if display_score >= 90: st.success("Excellent potential outcome! 🎉"); st.balloons()
                elif display_score >= 75: st.success("Good potential outcome.")
                elif display_score >= 60: st.info("Fair potential outcome.")
                else: st.warning("Potential challenges indicated.")

            except Exception as e:
                st.error(f"An error occurred during model prediction: {e}")
        else:
            st.error("Prediction could not be made due to preprocessing errors.")


st.markdown("---")
st.caption("Disclaimer: This predictive model provides an estimate based on patterns in the data it was trained on.")