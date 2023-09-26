import streamlit as st
from util import load_model
from lp_detection import lp_detect
from align_lp import align_lp
from lp_recognition import lp_recognition
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.write("# Vietnamese LP Recognition")

lp_detect_model = load_model("model/best_lp_detect.pt")
corner_detect_model = load_model("model/best_corner_detect.pt")
lp_recognition_model = load_model("model/best_lp_recognition.pt")


@st.cache_data
def show_recognition(
    img, 
    lp_detect_model=lp_detect_model, 
    corner_detect_model=corner_detect_model, 
    lp_recognition_model=lp_recognition_model, 
    mode="hough"
):
    # img = cv2.imread(img_path)
    lps, coords, img_lp_detect = lp_detect(img, lp_detect_model, save=False)
    contents = []
    img_copy = img.copy()
    if len(lps) != 0:
        for lp in lps:
            lp_align = align_lp(mode, lp, corner_detect_model, save=False)
            lp_char = lp_recognition(lp_align, lp_recognition_model)
            contents.append(lp_recognition(lp_align, lp_recognition_model))
        for (coord, content) in zip(coords, contents):
            cv2.rectangle(img_copy, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
            cv2.putText(img_copy, content, (coord[0], coord[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), lp_char

image_input = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])

if image_input is not None:

    image_numpy = cv2.imdecode(np.frombuffer(image_input.read(), np.uint8), 1)

    left, right = st.columns(2)
    left.write("input image")
    left.image(image_numpy, channels="BGR")

    right.write("Detecting...")
    lp_detect_image, lp_char = show_recognition(img = image_numpy)
    right.image(lp_detect_image)
    # st.image(lp_detect_image)
