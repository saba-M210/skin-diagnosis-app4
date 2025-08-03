import streamlit as st
from PIL import Image

# Set page config and remove sidebar
st.set_page_config(page_title="Skin AI System", layout="wide")

# Hide the Streamlit sidebar and footer using custom CSS
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            right: 2rem;
        }
        footer {
            visibility: hidden;
        }
        .main {
            background-color: #ffe6f0;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #000000;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 80px;
            margin-top: 20px;
        }
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .stButton > button {
            background-color: #dcdcff;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.75em 2em;
            font-size: 16px;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# --- UI layout ---
st.markdown("<div class='title'>Welcome</div>", unsafe_allow_html=True)

# Centered image + upload
st.markdown("<div class='centered'>", unsafe_allow_html=True)

# # Replace with your icon image file if needed
# st.image("icon_placeholder.png", width=200)

uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.session_state["user_image"] = uploaded_image

# --- Buttons Section ---
st.markdown("<div class='button-container'>", unsafe_allow_html=True)    
    
col1, col2 = st.columns(2)
with col1:
    if st.button("Skincare Recommendation"):
        st.session_state["selected_service"] = "skincare"
        st.switch_page("pages/skincare.py")

with col2:
    if st.button("Skin Disease Diagnosis"):
        st.session_state["selected_service"] = "diagnosis"
        st.switch_page("pages/diagnosis.py")

st.markdown("</div>", unsafe_allow_html=True)
