import time
import datetime
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array, smart_resize
from keras.metrics import top_k_categorical_accuracy
import plotly.graph_objects as go
from fpdf import FPDF
import io

@tf.keras.utils.register_keras_serializable()
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

@tf.keras.utils.register_keras_serializable()
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# Load model and cache it
@st.cache_resource
def load_model_once():
    model = load_model('inceptinv1.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', top_3_accuracy, top_2_accuracy])
    return model

model = load_model_once()

# Cache the prediction function
@st.cache_data
def cached_predict(_image):
    img = img_to_array(_image)
    img = img / 255.0
    img = smart_resize(img, (380, 380))  # Resize the image
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    return predictions

# Cache the PDF generation function
@st.cache_data
def generate_pdf(user_name, user_location, gender, date_of_birth, lesion_area, current_time, _image, predictions, top_3_classes, top_3_probs):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, 'Skin Cancer Classification Report', ln=True, align='C')
    pdf.ln(5)

    # User details
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Name: {user_name}", ln=True)
    pdf.cell(200, 10, f"Location: {user_location}", ln=True)
    pdf.cell(200, 10, f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, f"Date of Birth: {date_of_birth}", ln=True)
    pdf.cell(200, 10, f"Lesion Area: {lesion_area}", ln=True)
    pdf.cell(200, 10, f"Time of Classification: {current_time}", ln=True)
    pdf.ln(5)

    # Prediction results
    pdf.cell(200, 10, 'Prediction Results:', ln=True)
    for cls, prob in zip(top_3_classes, top_3_probs):
        pdf.cell(200, 10, f"{cls}: {prob:.2f}", ln=True)
    pdf.ln(5)

    # Add image to PDF
    image_path = "temp_image.png"
    _image.save(image_path)
    pdf.image(image_path, x=10, y=None, w=100)
    pdf.ln(10)  # Space after the image

    # Disclaimer
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This is a demo product for educational purposes only. "
                          "The classification results are for demonstration purposes and may not be accurate. "
                          "Please consult a medical professional for diagnosis and treatment.", align='L')
    pdf.ln(5)

    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, 'Developed by Sumanth Nimmagadda, Kiran Alex Challagiri, and Bakka Samuel Abhishek.', ln=True, align='C')
    pdf.cell(0, 10, 'For any queries, contact sumanthnimmagadda5667@gmail.com', ln=True, align='C')

    # Generate PDF in memory
    pdf_output = io.BytesIO()
    pdf_str = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_str)
    pdf_output.seek(0)  # Rewind the BytesIO object to the beginning

    return pdf_output

# Class descriptions
classes_description = {
    'akiec': "Actinic keratosis (AK), a precancerous skin lesion caused by prolonged exposure to UV radiation.",
    'bcc': "Basal cell carcinoma (BCC), the most common skin cancer, originating in basal cells of the epidermis.",
    'bkl': "Benign keratosis (BKL), a non-cancerous skin growth commonly found in older adults.",
    'df': "Dermatofibroma (DF), a benign skin tumor arising from fibrous tissue in the dermis.",
    'mel': "Melanoma (MEL), a malignant skin cancer originating from melanocytes, the skin's pigment-producing cells.",
    'nv': "Nevus (NV), commonly referred to as a mole, a benign proliferation of melanocytes.",
    'vasc': "Vascular lesion (VASC), conditions characterized by abnormalities in skin blood vessels."
}

# Demo product note
demo_note = "This is a demo product for educational purposes only. The classification results are for demonstration purposes and may not be accurate. Please consult a medical professional for diagnosis and treatment."

# Display class descriptions
st.sidebar.header('About Skin Cancer Classification')
for class_name, description in classes_description.items():
    st.sidebar.markdown(f"**{class_name.upper()}**: {description}")
    st.sidebar.markdown("---")

# Display demo product note
st.sidebar.markdown(demo_note)

st.title('Skin Cancer Classification App')

# User input
user_name = st.text_input("Enter your name", placeholder="Full Name")
user_location = st.text_input("Enter your location", placeholder='Country')
current_time = st.text_input("Current time and date", str(datetime.datetime.now()))
date_of_birth = st.date_input("Enter your date of birth", min_value=datetime.date(1965, 1, 1), max_value=datetime.date.today())
lesion_area = st.selectbox("Select lesion area", ["Head", "Torso", "Arm", "Leg", "Other"])
gender = st.selectbox("Select your gender", ["Male", "Female", "Other"])

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        with st.spinner("Analyzing the Lesion..."):
            time.sleep(1)  # Simulate waiting for verification and loading steps
            predictions = cached_predict(image)

        # Define your classes (if you haven't already)
        classes = {
            0: 'akiec',  # actinic keratosis
            1: 'bcc',  # basal cell carcinoma
            2: 'bkl',  # benign keratosis
            3: 'df',  # dermatofibroma
            4: 'mel',  # melanoma
            5: 'nv',  # nevus
            6: 'vasc'  # vascular lesion
        }

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_probs = predictions[0][top_3_indices]
        top_3_classes = [classes[i] for i in top_3_indices]

        # Display user info
        st.write(f"Hello, {user_name} from {user_location}!")

        # Display prediction results
        st.write("Prediction Results:")
        for cls, prob in zip(top_3_classes, top_3_probs):
            st.write(f"{cls}: {prob:.2f}")

        # Bar plot of probabilities
        fig = go.Figure(data=[go.Bar(x=top_3_classes, y=top_3_probs)])
        fig.update_layout(title='Class Probabilities', xaxis_title='Classes', yaxis_title='Probability')
        st.plotly_chart(fig)

        # Generate PDF
        pdf_output = generate_pdf(user_name, user_location, gender, date_of_birth, lesion_area, current_time, image, predictions, top_3_classes, top_3_probs)

        # Add PDF download button
        st.download_button(
            label="Download Report as PDF",
            data=pdf_output,
            file_name="skin_cancer_classification_report.pdf",
            mime="application/pdf"
        )

# Footer with developer names and LinkedIn link
footer = "Developed by Sumanth Nimmagadda, Kiran Alex Challagiri, and Bakka Samuel Abhishek. Connect with us on [LinkedIn](https://www.linkedin.com/in/sumanth-nimmagadda-472455221/)."
st.markdown("---")
st.markdown(footer)
st.warning("Disclaimer: This is a demo product for educational purposes only. The classification results are for demonstration purposes and may not be accurate. Please consult a medical professional for diagnosis and treatment.", icon='⚠️')
