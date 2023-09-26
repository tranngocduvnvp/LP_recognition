import streamlit as st
from util import load_model
from lp_detection import lp_detect
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.write("# Vietnamese LP Detection")

lp_detect_model = load_model("model/best_lp_detect.pt")


image_input = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])


if image_input is not None:

    image_numpy = cv2.imdecode(np.frombuffer(image_input.read(), np.uint8), 1)

    left, right = st.columns(2)
    left.write("input image")
    left.image(image_numpy, channels="BGR")

    right.write("Detection...")
    lps, coords, img_lp_detect = lp_detect(image_numpy, lp_detect_model, save=False) 
    right.image(img_lp_detect, channels="BGR")

    
