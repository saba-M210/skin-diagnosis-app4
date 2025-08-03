import streamlit as st

# Hide sidebar and apply background
st.set_page_config(page_title="Skincare Recommendation", layout="wide")

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
        .stButton > button {
            background-color: #dcdcff;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 1.5em;
            font-size: 16px;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# Page content
st.title("Skin Care Recommendation")

# st.write("This page will ask for user preferences and return personalized product recommendations.")

age = st.number_input("Enter your age", min_value=10, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
product_type = st.selectbox("Product Type", ["Cream", "Serum", "Cleanser"])
budget = st.number_input("Enter your budget", min_value=0)

if st.button("Get Recommendations"):
    st.write("Skin type: Normal (example)")
    st.write("Recommended Products:")
    st.write("- Product A: $10")
    st.write("- Product B: $15")
