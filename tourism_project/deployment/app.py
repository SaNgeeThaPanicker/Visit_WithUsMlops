import streamlit as st
import pickle
import pandas as pd
from huggingface_hub import hf_hub_download
import os

# Page config
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Wellness Tourism Package Predictor")
st.markdown("Fill in the customer details below to predict whether they will purchase the package.")

# Load model from Hugging Face
@st.cache_resource
def load_model():
    token = os.environ.get("HF_TOKEN")
    
    # Show clear error if token is missing
    if not token:
        st.error("HF_TOKEN secret is not set. Go to Space Settings → Secrets and add HF_TOKEN.")
        st.stop()
    
    model_path = hf_hub_download(
        repo_id="SANGU19/tourism-model",
        filename="best_model.pkl",
        token=token
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="SANGU19/tourism-model",
        filename="best_model.pkl",
        token=os.environ.get("HF_TOKEN")
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Input form
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=50000)
    num_persons_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_trips = st.number_input("Number of Trips Per Year", min_value=0, max_value=20, value=3)
    num_children_visiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

with col2:
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    passport = st.selectbox("Passport", ["Yes", "No"])
    own_car = st.selectbox("Own Car", ["Yes", "No"])

# Encode categorical inputs — sorted alphabetically to match LabelEncoder behavior
def encode(value, categories):
    categories_sorted = sorted(categories)
    return categories_sorted.index(value)

# Predict button
if st.button("Predict", use_container_width=True):

    input_data = {
        "Age": age,
        "TypeofContact": encode(type_of_contact, ["Company Invited", "Self Inquiry"]),
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": encode(occupation, ["Free Lancer", "Salaried", "Self Employed"]),
        "Gender": encode(gender, ["Female", "Male"]),
        "NumberOfPersonVisiting": num_persons_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": encode(product_pitched, ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]),
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": encode(marital_status, ["Divorced", "Married", "Single"]),
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": encode(designation, ["AVP", "Executive", "Manager", "Senior Manager", "VP"]),
        "MonthlyIncome": monthly_income
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Customer is **likely to purchase** the Wellness Package  \nConfidence: {probability:.1%}")
    else:
        st.error(f"❌ Customer is **unlikely to purchase** the Wellness Package  \nConfidence: {1 - probability:.1%}")