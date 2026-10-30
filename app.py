
import streamlit as st
from model import predict_performance

#Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

#Application title
st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's expected exam score "
    "using Machine Learning."
)

st.divider()

# Input section
st.header("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Daily Study Hours",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance Percentage",
        min_value=40,
        max_value=100,
        value=80
    )

with col2:
    previous_marks = st.slider(
        "Previous Exam Marks",
        min_value=30,
        max_value=100,
        value=70
    )

    sleep_hours = st.slider(
        "Daily Sleep Hours",
        min_value=4.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )

st.divider()

#Prediction button
if st.button("Predict Student Performance"):
    result = predict_performance(
        study_hours,
        attendance,
        previous_marks,
        sleep_hours
    )

    st.subheader("Prediction Result")

    st.metric(
        label="Expected Exam Score",
        value=f"{result}/100"
    )

    if result >= 75:
        st.success("Excellent expected performance!")
    elif result >= 50:
        st.info("Good potential. Keep improving!")
    else:
        st.warning("More preparation may be helpful.")

    st.caption(
        "This is an educational demonstration using "
        "synthetic sample data, not a real student assessment."
    )

st.divider()

st.caption("Powered by Python, Scikit-learn and Streamlit")
