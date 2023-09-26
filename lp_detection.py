import cv2
import torch
from pathlib import Path
import os
import argparse
from util import load_model
import streamlit as st

@st.cache_data
def lp_detect(img, _model, save_path='', save=True):
    name = ''
    if save:
        if save_path == '':
            return 'Please set save path'
        else:
            name = Path(img_path).stem
    img_copy = img.copy()
    results = _model(img, verbose=False)

    lp_list = []
    coord_list = []
    for result in results:
        if len(result.boxes.xyxy)==0:
            break
        for coord in result.boxes.xyxy:
            coord = coord.type(torch.int32).numpy()
            lp = img[coord[1]:coord[3], coord[0]:coord[2]]
            lp_list.append(lp)
            coord_list.append(coord)
            img_lp_detect = cv2.rectangle(img_copy, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
            if save == True:
                cv2.imwrite(os.path.join(save_path, '{}-[{},{},{},{}].jpg'.format(name, coord[0], coord[1], coord[2], coord[3])), lp)
            
    if save == True:
        cv2.imwrite(os.path.join(save_path, '{}-lp_detection_result.jpg'.format(name)), img_copy)
    return lp_list, coord_list, img_lp_detect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align License Plate')
    parser.add_argument('-img_path', '--img_path', type=str, help='lp img path for alignment', required=True)
    parser.add_argument('-model_path', '--model_path', type=str, help='model path for lp detection', default = 'model/best_lp_detect.pt')
    parser.add_argument('-save_path', '--save_path', type=str, help='saved path for lp aligment', required=True)
    args = parser.parse_args()

    img_path = args.img_path
    img = cv2.imread(img_path)
    save_path = args.save_path
    model = load_model(args.model_path)
    lp_detect(img, model, save_path)