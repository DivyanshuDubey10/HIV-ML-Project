import streamlit as st
import joblib
import pandas as pd


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/hiv_rf_model.joblib")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HIV Infection Classification",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(" HIV Infection Classification System")

st.write(
    "A machine learning application using the ACTG175 "
    "clinical-trial dataset and a tuned Random Forest classifier."
)

st.warning(
    "⚠️ Educational and research use only. "
    "This model is not clinically validated and must not be used "
    "to diagnose HIV infection or make medical decisions."
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header(" Patient Information")

st.info(
    "Enter demographic information recorded for the individual "
    "in the research dataset."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=100,
        value=30,
        help="Age of the individual at the beginning of the study."
    )

with col2:
    wtkg = st.number_input(
        "Weight (kg)",
        min_value=0.0,
        max_value=300.0,
        value=60.0,
        help="Body weight in kilograms measured at baseline."
    )


col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        help="Gender recorded in the original clinical-trial dataset."
    )

with col4:
    race = st.selectbox(
        "Race",
        options=["White", "Non-white"],
        help="Race category recorded in the original dataset."
    )


# ============================================================
# CLINICAL INFORMATION
# ============================================================

st.header(" Clinical Information")

st.info(
    "These fields describe clinical and behavioral characteristics "
    "recorded in the ACTG175 research dataset."
)

col1, col2 = st.columns(2)

with col1:
    hemo = st.selectbox(
        "Hemophilia",
        options=["No", "Yes"],
        help="Indicates whether the individual had hemophilia."
    )

with col2:
    homo = st.selectbox(
        "History of homosexual activity",
        options=["No", "Yes"],
        help="Historical behavioral variable recorded in the original study."
    )


col3, col4 = st.columns(2)

with col3:
    drugs = st.selectbox(
        "History of intravenous drug use",
        options=["No", "Yes"],
        help="Indicates whether a history of intravenous drug use was recorded."
    )

with col4:
    symptom = st.selectbox(
        "Symptoms",
        options=["Asymptomatic", "Symptomatic"],
        help="Whether the individual was recorded as symptomatic."
    )


karnof = st.number_input(
    "Karnofsky Performance Score",
    min_value=0,
    max_value=100,
    value=90,
    help=(
        "A performance-status score ranging from 0 to 100. "
        "Higher values generally indicate better functional status."
    )
)


# ============================================================
# TREATMENT HISTORY
# ============================================================

st.header(" Treatment History")

st.info(
    "These fields describe treatment assignment and previous "
    "antiretroviral treatment history in the clinical trial."
)

trt_label = st.selectbox(
    "Treatment group",
    options=[
        "ZDV only",
        "ZDV + ddI",
        "ZDV + Zalcitabine",
        "ddI only"
    ],
    help=(
        "Treatment group assigned in the ACTG175 clinical trial."
    )
)


col1, col2 = st.columns(2)

with col1:
    oprior = st.selectbox(
        "Previous non-ZDV antiretroviral therapy",
        options=["No", "Yes"],
        help=(
            "Whether previous non-ZDV antiretroviral therapy "
            "was recorded before the study treatment."
        )
    )

with col2:
    z30 = st.selectbox(
        "ZDV use during previous 30 days",
        options=["No", "Yes"],
        help=(
            "Whether ZDV (zidovudine) was used during the "
            "30 days before treatment initiation."
        )
    )


preanti = st.number_input(
    "Days of antiretroviral therapy before study",
    min_value=0,
    value=0,
    help=(
        "Number of days of antiretroviral therapy received "
        "before the study period."
    )
)


col1, col2 = st.columns(2)

with col1:
    str2_label = st.selectbox(
        "Antiretroviral treatment experience",
        options=["Naive", "Experienced"],
        help=(
            "Indicates whether the individual was antiretroviral "
            "treatment-naive or treatment-experienced."
        )
    )

with col2:
    strat_label = st.selectbox(
        "Antiretroviral history category",
        options=[
            "Antiretroviral naive",
            ">1 to 52 weeks of prior therapy",
            ">52 weeks of prior therapy"
        ],
        help=(
            "Stratification based on the amount of prior "
            "antiretroviral treatment."
        )
    )


col1, col2 = st.columns(2)

with col1:
    treat = st.selectbox(
        "Treatment indicator",
        options=["ZDV only", "Other therapies"],
        help=(
            "Binary treatment indicator used in the original dataset."
        )
    )

with col2:
    offtrt = st.selectbox(
        "Went off treatment before approximately 96 weeks",
        options=["No", "Yes"],
        help=(
            "Indicates whether the individual went off treatment "
            "before approximately 96 weeks."
        )
    )


# ============================================================
# LABORATORY INFORMATION
# ============================================================

st.header(" Laboratory Information")

st.info(
    "Enter the CD4 and CD8 T-cell measurements recorded in "
    "the research dataset."
)

col1, col2 = st.columns(2)

with col1:
    cd40 = st.number_input(
        "CD4 count at baseline",
        min_value=0,
        value=500,
        help="CD4 T-cell count measured at baseline."
    )

with col2:
    cd420 = st.number_input(
        "CD4 count at approximately 20 weeks",
        min_value=0,
        value=500,
        help="CD4 T-cell count measured approximately 20 weeks later."
    )


col3, col4 = st.columns(2)

with col3:
    cd80 = st.number_input(
        "CD8 count at baseline",
        min_value=0,
        value=1000,
        help="CD8 T-cell count measured at baseline."
    )

with col4:
    cd820 = st.number_input(
        "CD8 count at approximately 20 weeks",
        min_value=0,
        value=1000,
        help="CD8 T-cell count measured approximately 20 weeks later."
    )


# ============================================================
# TRIAL FOLLOW-UP INFORMATION
# ============================================================

st.header(" Trial Follow-up Information")

st.info(
    "These variables come from the longitudinal clinical-trial "
    "records used by the machine learning model."
)

time = st.number_input(
    "Time to failure/censoring (days)",
    min_value=0,
    value=100,
    help=(
        "Time recorded in days to the study's failure or "
        "censoring event."
    )
)


# ============================================================
# CONVERT HUMAN-READABLE VALUES TO DATASET ENCODINGS
# ============================================================

# Gender
gender_value = {
    "Female": 0,
    "Male": 1
}[gender]


# Race
race_value = {
    "White": 0,
    "Non-white": 1
}[race]


# Binary variables
hemo_value = {
    "No": 0,
    "Yes": 1
}[hemo]

homo_value = {
    "No": 0,
    "Yes": 1
}[homo]

drugs_value = {
    "No": 0,
    "Yes": 1
}[drugs]

oprior_value = {
    "No": 0,
    "Yes": 1
}[oprior]

z30_value = {
    "No": 0,
    "Yes": 1
}[z30]

symptom_value = {
    "Asymptomatic": 0,
    "Symptomatic": 1
}[symptom]

offtrt_value = {
    "No": 0,
    "Yes": 1
}[offtrt]


# Treatment group
trt_value = {
    "ZDV only": 0,
    "ZDV + ddI": 1,
    "ZDV + Zalcitabine": 2,
    "ddI only": 3
}[trt_label]


# Antiretroviral experience
str2_value = {
    "Naive": 0,
    "Experienced": 1
}[str2_label]


# Antiretroviral history stratification
strat_value = {
    "Antiretroviral naive": 1,
    ">1 to 52 weeks of prior therapy": 2,
    ">52 weeks of prior therapy": 3
}[strat_label]


# Treatment indicator
treat_value = {
    "ZDV only": 0,
    "Other therapies": 1
}[treat]


# ============================================================
# CREATE MODEL INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([{
    "time": time,
    "trt": trt_value,
    "age": age,
    "wtkg": wtkg,
    "hemo": hemo_value,
    "homo": homo_value,
    "drugs": drugs_value,
    "karnof": karnof,
    "oprior": oprior_value,
    "z30": z30_value,
    "preanti": preanti,
    "race": race_value,
    "gender": gender_value,
    "str2": str2_value,
    "strat": strat_value,
    "symptom": symptom_value,
    "treat": treat_value,
    "offtrt": offtrt_value,
    "cd40": cd40,
    "cd420": cd420,
    "cd80": cd80,
    "cd820": cd820
}])


# ============================================================
# PREDICTION
# ============================================================

st.header(" Model Prediction")

if st.button("Predict Infection", type="primary"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Infected")
    else:
        st.success("Prediction: Not Infected")

    st.metric(
        "Model probability for class 1",
        f"{probability:.2%}"
    )

    st.caption(
        "This probability is the model's estimated probability "
        "for class 1 in the dataset. It is not a clinical probability "
        "or medical diagnosis."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About the Machine Learning Model"):

    st.write(
        "This application uses a tuned Random Forest classifier "
        "trained on the ACTG175 HIV clinical-trial dataset."
    )

    st.write("Final evaluation metrics:")

    st.write("- Test Accuracy: 89.72%")
    st.write("- 5-Fold Cross-Validation Accuracy: 89.07%")
    st.write("- ROC-AUC: 0.92447")
