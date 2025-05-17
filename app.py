import streamlit as st
from PIL import Image
from transformers import pipeline
from google import genai
import re
import os

api_key = st.secrets["api_keys"]["gemini"]
genai_client = genai.Client(api_key=api_key)


# Load the classifier model
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="nateraw/vit-base-food101")
with st.spinner("Loading Model, Please Wait"):
    classifier = load_classifier()
st.title("🍽️ Food Image → Dish, Ingredients & Calories")

# Upload image
uploaded_image = st.file_uploader("Upload a food image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp file
    temp_path = "temp.jpg"
    image.save(temp_path)

    # Run classification
    results = classifier(temp_path)
    top_labels = [res["label"] for res in results][:1]


    if not top_labels:
        st.warning("⚠️ No confident food labels found.")
    else:
        st.markdown(f"**🧠 Top labels:** {', '.join(top_labels)}")

        # Prompt Gemini
        prompt = f"""
Given these possible food labels: {', '.join(top_labels)},
1. List only the key ingredients (3–6 items max).
2. Give calories per ingredient per serving.
3. Estimate total calories per serving (as a number only).
4. Dish Name based on label recieved.

Respond in this format:
Dish Name: <name>
Ingredients: <comma-separated>
Calories Per Ingredient: <comma separated>
Total Calories Per Serving: <number>
"""
        with st.spinner("Giving Calorie Info ...."):
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )

        output = response.text.strip()
        # st.markdown("### 🧠 Gemini Output")
        # st.text(output)

        # Extract clean data
        dish = re.search(r'Dish Name:\s*(.+)', output)
        ingredients = re.search(r'Ingredients:\s*(.+)', output)
        cal_per_ing = re.search(r'Calories Per Ingredient:\s*(.+)', output)
        total_cal = re.search(r'Total Calories:\s*(\d+)', output)

        st.markdown("### ✅ Final Result")
        st.write("🍽️ **Dish Name:**", dish.group(1).strip() if dish else "N/A")
        st.write("🧾 **Ingredients:**", ingredients.group(1).strip() if ingredients else "N/A")
        st.write("🔥 **Calories Per Ingredient:**", cal_per_ing.group(1).strip() if cal_per_ing else "N/A")
        st.write("🔥 **Total Calories:**", total_cal.group(1).strip() if total_cal else "N/A")
