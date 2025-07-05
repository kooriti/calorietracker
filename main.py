import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import pandas as pd

@st.cache_resource
def init_model():
    processor = AutoImageProcessor.from_pretrained("prithivMLmods/Food-101-93M")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Food-101-93M")
    return processor, model

processor, model = init_model()

@st.cache_data
def load_calorie_data(path="food_calorie.csv"):
    df = pd.read_csv(path)
    df.drop_duplicates(subset="label", keep="first", inplace=True)
    df.set_index("label", inplace=True)
    return df

calorie_map = load_calorie_data()

st.set_page_config(page_title="AI Food Calorie Estimator", layout="centered")
st.title("🍽️ AI Food Calorie Estimator")
st.write("Upload a food image and get full nutritional information!")

uploaded = st.file_uploader("Upload a food image", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Estimate Nutrition"):
        with st.spinner("Analyzing..."):
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = logits.argmax(-1).item()
            pred_label = model.config.id2label[pred_id]

            normalized_label = pred_label.replace("_", " ").lower()
            matched_row = None
            matched_label = None
            for label in calorie_map.index:
                if label.lower() == normalized_label:
                    matched_row = calorie_map.loc[label]
                    matched_label = label
                    break

            st.session_state['matched_row'] = matched_row
            st.session_state['matched_label'] = matched_label

    if 'matched_row' in st.session_state and st.session_state['matched_row'] is not None:
        matched_row = st.session_state['matched_row']
        matched_label = st.session_state['matched_label']

        amt = st.number_input("How many grams did you eat?", value=100, min_value=1, step=1)

        scaled = (matched_row * (amt / matched_row["weight"])).round(2)

        st.subheader(f"🥘 Detected: **{matched_label}**")
        st.markdown("### 🍴 Estimated Nutrition Facts:")
        for nutrient, value in scaled.items():
            if nutrient != "weight":
                st.write(f"**{nutrient.capitalize()}**: {value}")
    elif 'matched_row' in st.session_state and st.session_state['matched_row'] is None:
        st.warning("Sorry, no nutrition data found for the detected food.")
