import streamlit as st
from util import load_model
from lp_recognition import lp_recognition
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.write("# Vietnamese LP Recognition")

lp_recognition_model = load_model("model/best_lp_recognition.pt")


image_input = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])


if image_input is not None:

    image_numpy = cv2.imdecode(np.frombuffer(image_input.read(), np.uint8), 1)
    st.write("input image")
    st.image(image_numpy, channels="BGR")
    lp_char = lp_recognition(image_numpy, lp_recognition_model) 
    st.write(f"Recognition: {lp_char}")
    

    
