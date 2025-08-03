import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import joblib
from transformers import AutoImageProcessor, SwinModel
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
import io
from huggingface_hub import hf_hub_download

# --------------------- CONFIGURATION ---------------------
# Constants
UNIQUE_CLASSES = ['Actinic_keratosis', 'Basal_cell_carcinoma', 'Benign_keratosis', 
                  'Dermatofibroma', 'Melanocytic_nevus', 'Melanoma', 
                  'Squamous_cell_carcinoma', 'Vascular_lesion']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "Sabamo/skin-diagnosis-model"  # Replace with your actual HF repo


SWIN_MODEL_PATH = hf_hub_download(repo_id="Sabamo/skin-diagnosis-model", filename="swin.pth")
RF_MODEL_PATH = hf_hub_download(repo_id="Sabamo/skin-diagnosis-model", filename="rf.pkl")

# Set your API key
os.environ['OPENROUTER_API_KEY'] = "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"

# --------------------- STYLING ---------------------
st.set_page_config(page_title="Diagnosis", layout="centered")
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        footer { visibility: hidden; }
        .main { background-color: #ffe6f0; }
        .title { text-align: center; font-size: 28px; font-weight: bold; color: black; margin-bottom: 30px; }
        .uploaded-msg { margin-top: 10px; font-weight: bold; }
        .stButton > button {
            background-color: #dcdcff; color: black; font-weight: bold;
            border-radius: 10px; padding: 0.5em 1.5em; font-size: 16px;
        }
        input, select { background-color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# --------------------- Load Models ---------------------
import torch.nn as nn  # Add this if not already

# --------------------- Load Models ---------------------
@st.cache_resource
def load_swin_model():
    class OptimizedSwinClassifier(nn.Module):
        def __init__(self, num_classes=8, dropout_rate=0.3):
            super(OptimizedSwinClassifier, self).__init__()
            self.backbone = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224-in22k")
            hidden_size = self.backbone.config.hidden_size

            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate // 2),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.BatchNorm1d(hidden_size // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate // 4),
                nn.Linear(hidden_size // 4, 8)
            )

        def forward(self, pixel_values):
            outputs = self.backbone(pixel_values=pixel_values)
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            return self.classifier(pooled_output)

        def extract_features(self, pixel_values):
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
                pooled_output = outputs.last_hidden_state.mean(dim=1)
                return pooled_output.detach().cpu().numpy()

    swin_path = hf_hub_download(repo_id=REPO_ID, filename="swin.pth", repo_type="model")
    model = OptimizedSwinClassifier()
    model.load_state_dict(torch.load(swin_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window7-224-in22k")
    return model, processor


@st.cache_resource
def load_rf_model():
    import pickle
    import io
    rf_path = hf_hub_download(repo_id=REPO_ID, filename="rf.pkl", repo_type="model")

    try:
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        return rf_model
    except Exception as e:
        st.error(f"Error loading RF model: {e}")
        raise

# --------------------- Metadata & Features ---------------------
def create_user_metadata_input(age, gender, anatomic_site):
    sites = ['anterior torso', 'head/neck', 'lower extremity', 
             'oral/genital', 'palms/soles', 'posterior torso', 'upper extremity']
    ages = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-90']
    
    if age <= 10: age_range = '1-10'
    elif age <= 20: age_range = '11-20'
    elif age <= 30: age_range = '21-30'
    elif age <= 40: age_range = '31-40'
    elif age <= 50: age_range = '41-50'
    elif age <= 60: age_range = '51-60'
    elif age <= 70: age_range = '61-70'
    else: age_range = '71-90'

    metadata = {f'site_{s}': 0 for s in sites}
    metadata[f'site_{anatomic_site}'] = 1
    metadata['sex_female'] = 1 if gender.lower() == "female" else 0
    metadata['sex_male'] = 1 if gender.lower() == "male" else 0
    metadata.update({f'age_{a}': 0 for a in ages})
    metadata[f'age_{age_range}'] = 1
    return pd.DataFrame([metadata])

def combine_features_metadata(image_features, user_metadata_df):
    feature_cols = [f"feature_{i}" for i in range(len(image_features))]
    features_df = pd.DataFrame([image_features], columns=feature_cols)
    return pd.concat([user_metadata_df.reset_index(drop=True), features_df], axis=1)

def extract_image_features(image: Image.Image, model, processor):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    pooled_output = outputs.pooler_output.squeeze().cpu().numpy()
    return pooled_output

# --------------------- LangChain Agents ---------------------
class MedicalState(TypedDict):
    disease: str
    country: str
    disease_info: str
    hospital_list: str

def get_llm(model="gpt-3.5-turbo", temp=0.2):
    return ChatOpenAI(
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        model=model,
        temperature=temp
    )

def disease_info_agent(state: MedicalState) -> MedicalState:
    prompt = f"""
You are a helpful medical assistant.
Explain the skin disease "{state['disease']}" in 4–5 sentences and include:
1- symptoms
2- causes
3- treatments
"""
    llm = get_llm()
    result = llm.invoke([SystemMessage(content="You are a reliable, medically informed assistant."),
                         HumanMessage(content=prompt)])
    return {**state, "disease_info": result.content}

def hospital_finder_agent(state: MedicalState) -> MedicalState:
    prompt = f"""
List 5 hospitals or dermatology clinics in {state['country']} with:
- name, city, contact (if possible), short address.
Only list dermatology-relevant institutions.
"""
    llm = get_llm()
    result = llm.invoke([SystemMessage(content="You are a medical travel assistant."),
                         HumanMessage(content=prompt)])
    return {**state, "hospital_list": result.content}

def build_workflow():
    graph = StateGraph(MedicalState)
    graph.add_node("describe_disease", disease_info_agent)
    graph.add_node("find_hospitals", hospital_finder_agent)
    graph.set_entry_point("describe_disease")
    graph.add_edge("describe_disease", "find_hospitals")
    graph.add_edge("find_hospitals", END)
    return graph.compile()

def run_medical_assistant(disease, country):
    assistant = build_workflow()
    state = {
        "disease": disease.strip(),
        "country": country.strip(),
        "disease_info": "",
        "hospital_list": ""
    }
    result = assistant.invoke(state)
    return result["disease_info"], result["hospital_list"]

# --------------------- Streamlit UI ---------------------
st.markdown("<div class='title'>Skin Disease Diagnosis</div>", unsafe_allow_html=True)

age = st.number_input("Age:", min_value=1, max_value=100, value=30)
gender = st.selectbox("Gender:", ["Male", "Female"])
anatomic_site = st.selectbox("Affected Body Area:", [
    'anterior torso', 'head/neck', 'lower extremity', 'oral/genital',
    'palms/soles', 'posterior torso', 'upper extremity'
])
country = st.text_input("Country (for hospital lookup):", value="USA")

if st.button("Diagnose"):
    with st.spinner("Analyzing..."):
        swin_model, swin_processor = load_swin_model()
        rf_model = load_rf_model()

        image = Image.open(io.BytesIO(st.session_state["user_image"].getvalue())).convert("RGB")
        image_features = extract_image_features(image, swin_model, swin_processor)
        user_metadata = create_user_metadata_input(age, gender, anatomic_site)
        combined_input = combine_features_metadata(image_features, user_metadata)

        prediction = rf_model.predict(combined_input)[0]
        confidence = np.max(rf_model.predict_proba(combined_input))

        disease_info, hospital_list = run_medical_assistant(prediction, country)

    st.success(f"**Predicted Disease:** {prediction} (Confidence: {confidence:.2%})")
    st.markdown(f"### Disease Information\n{disease_info}")
    st.markdown(f"### Hospitals in {country}\n{hospital_list}")
