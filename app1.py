import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from fpdf import FPDF
import base64
from PIL import Image
import numpy as np
import pydicom
import cv2
from skimage.util import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
import tempfile
import shutil
import logging
from pathlib import Path
from ultralytics import YOLO

# --- Basic Setup ---
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging system initialized")

# --- Stage 1: Data from the Computer Vision Model ---
TB_CLASSIFICATION_DATA = {
    "Artefact": {"TB_Category": "No", "RISK": "NA"},
    "Bullous Changes": {"TB_Category": "No", "RISK": "NA"},
    "Bullous Lesions": {"TB_Category": "No", "RISK": "NA"},
    "Calcification": {"TB_Category": "No", "RISK": "NA"},
    "Cardiomegaly": {"TB_Category": "No", "RISK": "NA"},
    "Cavity": {"TB_Category": "Yes", "RISK": "Medium"},
    "Collapse": {"TB_Category": "No", "RISK": "NA"},
    "Consolidation": {"TB_Category": "Yes", "RISK": "Medium"},
    "Cystic Shadows": {"TB_Category": "No", "RISK": "NA"},
    "Effusion": {"TB_Category": "Yes", "RISK": "High"},
    "Elevated Diaphragm": {"TB_Category": "No", "RISK": "NA"},
    "Fibroproductive Lesion": {"TB_Category": "Yes", "RISK": "High"},
    "Fibrosis": {"TB_Category": "Yes", "RISK": "Medium"},
    "Flattened Diaphragm": {"TB_Category": "No", "RISK": "NA"},
    "Hilar Adenopathy": {"TB_Category": "Yes", "RISK": "High"},
    "Hilar Prominence": {"TB_Category": "No", "RISK": "NA"},
    "Hyperinflation": {"TB_Category": "No", "RISK": "NA"},
    "Increased Broncho-vascular Markings": {"TB_Category": "No", "RISK": "NA"},
    "Increased Lucency": {"TB_Category": "No", "RISK": "NA"},
    "Infiltrates": {"TB_Category": "Yes", "RISK": "Medium"},
    "Lymphangitis": {"TB_Category": "No", "RISK": "NA"},
    "Mass": {"TB_Category": "Yes", "RISK": "Low"},
    "Mediastinal Widening": {"TB_Category": "Yes", "RISK": "Low"},
    "Miliary nodules": {"TB_Category": "Yes", "RISK": "High"},
    "Nodules": {"TB_Category": "Yes", "RISK": "High"},
    "Normal": {"TB_Category": "No", "RISK": "NA"},
    "Pleural Plaque": {"TB_Category": "No", "RISK": "NA"},
    "Pleural Thickening": {"TB_Category": "No", "RISK": "NA"},
    "Pneumothorax": {"TB_Category": "No", "RISK": "NA"},
    "Prominent Pulmonary Artery": {"TB_Category": "No", "RISK": "NA"},
    "Reticulonodular Shadows": {"TB_Category": "Yes", "RISK": "Medium"},
    "Rib Erosion": {"TB_Category": "No", "RISK": "NA"},
    "Scoliosis": {"TB_Category": "No", "RISK": "NA"},
    "Tubular Heart": {"TB_Category": "No", "RISK": "NA"},
    "Vessel Pruning": {"TB_Category": "No", "RISK": "NA"}
}

def get_tb_info(class_name: str) -> dict | None:
    """Looks up the TB Category and RISK for a given class name."""
    logger.info(f"Looking up TB info for class: {class_name}")
    result = TB_CLASSIFICATION_DATA.get(class_name)
    if result is None:
        logger.warning(f"No TB info found for class: {class_name}")
    return result

# --- Stage 1.5: Image Processing ---
def process_image(uploaded_file, output_dir: str) -> tuple[str, str, str] | None:
    """Processes uploaded image and runs YOLO detection."""
    logger.info("Starting image processing")
    try:
        # Save uploaded file to a temporary path within the provided directory
        temp_dcm_path = os.path.join(output_dir, "temp_input.dcm")
        uploaded_file.seek(0)
        with open(temp_dcm_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file to: {temp_dcm_path}")

        # Process the DICOM file
        dicom = pydicom.dcmread(temp_dcm_path)
        img_array = dicom.pixel_array.astype(np.float32)
        
        img_norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        resized = cv2.resize(img_norm, (512, 512), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        white_balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(white_balanced, cv2.COLOR_BGR2GRAY)
        
        gray_float = img_as_float(gray)
        sigma = estimate_sigma(gray_float, channel_axis=None, average_sigmas=True)
        denoised = denoise_nl_means(gray_float, h=1.15 * sigma, fast_mode=True, patch_size=5, patch_distance=3)
        
        denoised_uint8 = (denoised * 255).astype(np.uint8)
        final_rgb = cv2.cvtColor(denoised_uint8, cv2.COLOR_GRAY2RGB)

        processed_path = os.path.join(output_dir, "processed_image.png")
        cv2.imwrite(processed_path, final_rgb)
        logger.info(f"Saved processed image to: {processed_path}")

        # --- YOLO INFERENCE ---
        logger.info("Running YOLO inference")
        # CORRECTED: Added 'r' to create a raw string for the Windows path
        model = YOLO(r"C:\Users\jaswa\NEUZENAI IT SOLUTIONS PVT LTD\NEUZENAI IT SOLUTIONS PVT LTD - Documents\AI4TG - Imaging Diagnostics Solution for TB\charan_coding_TB_USE_CASE\best.pt")
        
        results = model(processed_path)
        
        if not results or not results[0].boxes:
            logger.warning("YOLO model did not detect any objects.")
            # Default to "Normal" if no detection
            best_class_name = "Normal"
            annotated_img = final_rgb # No annotations to plot
        else:
            boxes = results[0].boxes
            conf_scores = boxes.conf.cpu().numpy()
            
            if len(conf_scores) == 0:
                logger.warning("YOLO detected boxes but with no confidence scores.")
                best_class_name = "Normal"
                annotated_img = final_rgb
            else:
                pred_classes = boxes.cls.cpu().numpy()
                class_names = model.names
                
                max_index = conf_scores.argmax()
                best_class_id = int(pred_classes[max_index])
                best_conf = conf_scores[max_index]
                best_class_name = class_names[best_class_id]
                logger.info(f"Best prediction -> Class: {best_class_name}, Confidence: {best_conf:.2f}")

                annotated_img = results[0].plot()

        annotated_path = os.path.join(output_dir, "annotated_image.png")
        cv2.imwrite(annotated_path, annotated_img)
        logger.info(f"Saved annotated image to: {annotated_path}")

        return processed_path, annotated_path, best_class_name

    except Exception as e:
        logger.error(f"Error in image processing: {str(e)}", exc_info=True)
        st.error(f"Error in image processing or YOLO inference: {e}")
        return None

# --- Stage 2: Gemini Integration ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    logger.error("GOOGLE_API_KEY not found. Please create a .env file.")
    st.error("GOOGLE_API_KEY not found. Please set it up in a .env file.")
else:
    try:
        genai.configure(api_key=google_api_key)
        logger.info("Google Generative AI configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI: {str(e)}", exc_info=True)
        st.error(f"Failed to configure Google Generative AI: {e}")
        google_api_key = None

def get_gemini_response(input_text: str) -> str:
    """Sends prompt to Gemini model and returns response."""
    if not google_api_key:
        logger.error("Attempted Gemini call without a valid API key.")
        return "AI model is not configured due to missing or invalid API key."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(input_text)
        if not response.text:
            logger.warning(f"Gemini returned an empty response. Finish reason: {response.candidates[0].finish_reason}")
            return f"AI model returned an empty response. This may be due to safety filters."
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
        return f"An error occurred while calling the AI model: {e}"

# --- Stage 3: PDF and UI ---
def sanitize_text(text):
    """Clean text for PDF generation."""
    return str(text).encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(finding, category, risk, result, output_dir: str) -> str | None:
    """Generates PDF report with diagnosis details."""
    logger.info("Creating PDF report")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Swass-AI Medical Recommendation", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, sanitize_text("Image Analysis Results:"), ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, f"- Detected Finding: {sanitize_text(finding)}\n- TB Category: {sanitize_text(category)}\n- Risk Level: {sanitize_text(risk)}",)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, sanitize_text("AI-Generated Medical Plan:"), ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, sanitize_text(result))

    output_path = os.path.join(output_dir, "Medical_Recommendation_Report.pdf")
    try:
        pdf.output(output_path)
        logger.info(f"PDF report saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save PDF file: {str(e)}", exc_info=True)
        st.error(f"Failed to save PDF file: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Swass-AI", page_icon="🏥", layout="centered")

st.image("https://1.bp.blogspot.com/-wE5O8JmxtrQ/W13WbcEb2WI/AAAAAAAAXLY/YoRjZdGkMSEiT17GhRMYGlSqH3pq4MQNgCLcBGAs/s1600/kanti+velugu+02.png", width=120)
st.markdown("<h2 style='text-align: center; color: #004085;'>NEUZENAI-TB-DETECTRON</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Chest X-ray Image (DCM format)", type=["dcm"])

submit = st.button("💊 Analyse Image and Generate AI INSIGHTS", disabled=(not uploaded_file))

if submit:
    with tempfile.TemporaryDirectory() as temp_output_dir:
        logger.info(f"Created temporary directory: {temp_output_dir}")
        try:
            with st.spinner("Processing image and running analysis..."):
                # Call the processing function
                result_tuple = process_image(uploaded_file, temp_output_dir)

                # --- ROBUST LOGIC: Check if processing was successful ---
                if result_tuple:
                    processed_image_path, annotated_path, detected_class = result_tuple
                    
                    st.image(annotated_path, caption='Annotated Chest X-ray', use_container_width=True)
                    
                    tb_info = get_tb_info(detected_class)

                    if tb_info:
                        tb_category = tb_info['TB_Category']
                        risk_level = tb_info['RISK']

                        st.markdown("### 🔎 Analysis Results")
                        st.write(f"**Detected Finding:** {detected_class}")
                        st.write(f"**TB Category:** {tb_category}")
                        st.write(f"**Risk Level:** {risk_level}")

                        # Create prompt for Gemini
                        prompt = f"""
                        You are a specialized medical advisory AI. Your task is to provide a standard medication and prevention plan, including dietary advice, based on a pre-diagnosed condition from a computer vision model analysis of a chest X-ray. DO NOT change the diagnosis or risk level provided.

                        Image Analysis Output:
                        - Detected Class: {detected_class}
                        - Is TB-Related Category: {tb_category}
                        - Assessed Risk Level: {risk_level}

                        Your Task:
                        If "Is TB-Related Category" is "No", state that "{detected_class}" is not typically associated with Tuberculosis and no TB-specific medication is required. Provide general lung health advice.

                        If "Is TB-Related Category" is "Yes", provide a structured response covering:
                        1. Recommended Medication Plan: Describe standard anti-tuberculosis therapy concepts (e.g., multi-drug regimen, intensive/continuation phases) based on the risk level. Do not name specific drugs.
                        2. Treatment Duration: State typical duration.
                        3. Dietary Advice: Give 3-4 bullet points on diet for TB patients.
                        4. Prevention and Management Advice: Give 3-4 bullet points on adherence, hygiene, and lifestyle.
                        5. Disclaimer: Include a clear disclaimer that this is AI-generated and a doctor must be consulted for diagnosis and treatment.
                        """
                        
                        with st.spinner("Generating AI recommendations..."):
                            gemini_result = get_gemini_response(prompt)
                        
                        st.markdown("### 🧾 AI-Generated Recommendation")
                        st.markdown(gemini_result)

                        # Generate PDF
                        pdf_path = create_pdf_report(detected_class, tb_category, risk_level, gemini_result, temp_output_dir)
                        if pdf_path and os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                b64_pdf = base64.b64encode(f.read()).decode()
                                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Medical_Recommendation_Report.pdf" style="display:inline-block;padding:0.5em 1em;background-color:#007bff;color:white;text-align:center;text-decoration:none;border-radius:0.25em;">📄 Download PDF Report</a>'
                                st.markdown(href, unsafe_allow_html=True)

                    else: # tb_info is None
                        st.error(f"Internal data error: No TB classification info found for the detected class '{detected_class}'.")
                
                else: # result_tuple is None
                    st.error("Image processing failed. Please check the uploaded file or logs for more details.")

        except Exception as e:
            logger.error(f"An unexpected error occurred in the main process: {str(e)}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")