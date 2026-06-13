import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("stress_model2.joblib")

st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-title {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
}

.sub-title {
    text-align: center;
    color: #AAAAAA;
    margin-bottom: 30px;
}

.metric-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e1e1e;
    text-align: center;
}

.stButton>button {
    width: 100%;
    height: 60px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="big-title">🧠 Student Stress Predictor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">AI Powered Mental Wellness Prediction System</p>',
    unsafe_allow_html=True
)

# Layout
col1, col2 = st.columns(2)

with col1:

    student_type = st.selectbox(
        "🎓 Student Type",
        ["college", "school"]
    )

    sleep_hours = st.slider(
        "😴 Sleep Hours",
        0.0, 12.0, 7.0
    )

    study_hours = st.slider(
        "📚 Study Hours",
        0.0, 12.0, 4.0
    )

    social_media_hours = st.slider(
        "📱 Social Media Hours",
        0.0, 12.0, 3.0
    )

with col2:

    attendance = st.slider(
        "🏫 Attendance (%)",
        0.0, 100.0, 75.0
    )

    exam_pressure = st.slider(
        "🔥 Exam Pressure",
        1, 10, 5
    )

    family_support = st.slider(
        "❤️ Family Support",
        1, 10, 5
    )

    month = st.selectbox(
        "📅 Month",
        list(range(1, 13))
    )

student_type_encoded = (
    0 if student_type == "college" else 1
)

if st.button("🚀 Predict Stress Level"):

    features = np.array([[
        student_type_encoded,
        sleep_hours,
        study_hours,
        social_media_hours,
        attendance,
        exam_pressure,
        family_support,
        month
    ]])

    prediction = model.predict(features)[0]

    try:
        probability = model.predict_proba(features)[0][1]
    except:
        probability = None

    st.divider()

    if prediction == 1:

        st.error(
            "⚠️ HIGH STRESS LEVEL DETECTED"
        )

    else:

        st.success(
            "✅ LOW STRESS LEVEL DETECTED"
        )

    if probability is not None:

        st.subheader("Prediction Confidence")

        st.progress(float(probability))

        st.metric(
            "Stress Probability",
            f"{probability*100:.2f}%"
        )

        if probability > 0.7:
            st.warning(
                "Student may be under significant stress."
            )

        elif probability > 0.4:
            st.info(
                "Moderate stress indicators detected."
            )

        else:
            st.success(
                "Stress indicators appear low."
            )

st.divider()

st.caption(
    "Built using Logistic Regression, Scikit-Learn and Streamlit"
)