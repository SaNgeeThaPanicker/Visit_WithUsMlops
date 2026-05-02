import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download

# ── Load model and encoders from Hugging Face Model Hub ──────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="SANGU19/tourism-model",
        filename="best_model.pkl",
        repo_type="model"
    )
    encoders_path = hf_hub_download(
        repo_id="SANGU19/tourism-dataset",
        filename="label_encoders.pkl",
        repo_type="dataset"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Tourism Package Purchase Predictor")
st.markdown("Predict whether a customer will purchase the Wellness Tourism Package.")

col1, col2 = st.columns(2)

with col1:
    age               = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income    = st.number_input("Monthly Income", min_value=0, value=50000)
    num_trips         = st.number_input("Number of Trips", min_value=0, value=3)
    city_tier         = st.selectbox("City Tier", [1, 2, 3])
    occupation        = st.selectbox("Occupation", encoders["Occupation"].classes_)
    gender            = st.selectbox("Gender", encoders["Gender"].classes_)

with col2:
    type_of_contact   = st.selectbox("Type of Contact", encoders["TypeofContact"].classes_)
    marital_status    = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
    designation       = st.selectbox("Designation", encoders["Designation"].classes_)
    num_followups     = st.number_input("Number of Followups", min_value=0, value=3)
    pitch_score       = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    passport          = st.selectbox("Has Passport", [0, 1])

num_persons   = st.number_input("Number of Persons Visiting", min_value=1, value=2)
num_children  = st.number_input("Number of Children Visiting", min_value=0, value=0)
property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
own_car       = st.selectbox("Owns Car", [0, 1])
product_pitched = st.selectbox("Product Pitched", encoders["ProductPitched"].classes_)
duration      = st.number_input("Duration of Pitch (mins)", min_value=0, value=15)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Age":                      age,
        "TypeofContact":            encoders["TypeofContact"].transform([type_of_contact])[0],
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration,
        "Occupation":               encoders["Occupation"].transform([occupation])[0],
        "Gender":                   encoders["Gender"].transform([gender])[0],
        "NumberOfPersonVisiting":   num_persons,
        "NumberOfFollowups":        num_followups,
        "ProductPitched":           encoders["ProductPitched"].transform([product_pitched])[0],
        "PreferredPropertyStar":    property_star,
        "MaritalStatus":            encoders["MaritalStatus"].transform([marital_status])[0],
        "NumberOfTrips":            num_trips,
        "Passport":                 passport,
        "OwnCar":                   own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation":              encoders["Designation"].transform([designation])[0],
        "MonthlyIncome":            monthly_income,
        "PitchSatisfactionScore":   pitch_score,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer is LIKELY to purchase the package (Confidence: {probability:.1%})")
    else:
        st.warning(f"❌ Customer is UNLIKELY to purchase the package (Confidence of No: {1-probability:.1%})")
