# 🎓 Student Exam Score Predictor

A simple web application built using Streamlit that predicts student exam scores. It utilizes a CatBoost regression model trained on data encompassing student demographics, academic habits, and lifestyle factors.

## ✨ Features

* Interactive web form for inputting student data using sliders, dropdowns, and number inputs.
* Predicts numerical exam scores based on the provided inputs.
* Applies necessary preprocessing and feature engineering consistent with model training.
* Clips the final predicted score to a realistic range (15-100).
* Provides a simple interpretation of the predicted score range.
* Utilizes a pre-trained CatBoost model loaded from saved artifacts.
* Features a simple, clean UI with a light green background.

## 🛠️ Model Details

* **Algorithm:** CatBoost Regressor (`catboost.CatBoostRegressor`)
* **Preprocessing:**
    * **Binary Features** (`part_time_job`, `extracurricular_participation`): Encoded as 1 for 'Yes'/True and 0 for 'No'/False. NaNs defaulted to 0.
    * **Categorical Features** (`gender`, `diet_quality`, `parental_education_level`, `internet_quality`): Handled internally by the CatBoost model. Missing values (`NaN`) during preprocessing were filled with the string `"Missing"` before training and are handled similarly for prediction inputs.
* **Feature Engineering:**
    * `wellbeing_index`: Created by applying Min-Max scaling (to range [0, 1]) to `sleep_hours`, `exercise_frequency`, and `mental_health_rating` based on assumed realistic ranges (`SCALING_RANGES` in `app.py`), and then summing these scaled values. Potential NaNs in components were imputed before scaling.
    * `study_consistency`: Calculated as `study_hours_per_day * (attendance_percentage / 100.0)`.
    * `total_screen_time`: Calculated as `netflix_hours + social_media_hours`.
    * The original numerical features used to create these engineered features (`sleep_hours`, `exercise_frequency`, `mental_health_rating`, `study_hours_per_day`, `attendance_percentage`, `netflix_hours`, `social_media_hours`) were **dropped** before training the final model.
* **Prediction Clipping:** Raw model output is clipped to the range \[15.0, 100.0] before being displayed.

## ⚙️ Setup Instructions

1.  **Clone the repository:**
    
2.  **Create and activate a Python virtual environment (Recommended):**
    

3.  **Install required dependencies:**
   
4.  **Ensure Model Artifacts are Present:**

## ⚠️ Disclaimer

This predictive model provides an estimate based on patterns in the data it was trained on. Actual student scores can vary significantly due to numerous factors not included in the model or the input data. It should be used for illustrative purposes only.