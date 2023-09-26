from ultralytics import YOLO
import streamlit as st


@st.cache_resource
def load_model(path):
    return YOLO(path) 