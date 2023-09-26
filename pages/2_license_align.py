import streamlit as st
from util import load_model
from align_lp import align_lp
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.write("# Vietnamese LP Alignment")

corner_detect_model = load_model("model/best_corner_detect.pt")


mode_align = st.selectbox("Align mode, hough or keypoint",
                           ("hough", "keypoint"))
image_input = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])


if image_input is not None:

    image_numpy = cv2.imdecode(np.frombuffer(image_input.read(), np.uint8), 1)

    left, right = st.columns(2)
    left.write("input image")
    left.image(image_numpy, channels="BGR")

    right.write("Align...")
    image_align = align_lp(mode_align, image_numpy, corner_detect_model, save=False) 
    right.image(image_align, channels="BGR")

    
