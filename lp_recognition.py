import cv2
import torch
import argparse
import math
from util import load_model
import streamlit as st

def group_lines(points):
    # Calculate the average y-coordinate
    average_y = sum(y for _, _, y in points) / len(points)

    # Group points into two lines based on y-coordinate
    line1 = []
    line2 = []

    for class_idx, x, y in points:
        if y <= average_y:
            line1.append((class_idx, x, y))
        else:
            line2.append((class_idx, x, y))

    return line1, line2

characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V',  'X', 'Y', 'Z',  '0' ]

def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

@st.cache_data
def lp_recognition(img, _model):
    results = _model(img, verbose=False)
    lp_content = ''
    for result in results:
        new_data = []
        for data in result.boxes.data:
            data = data.type(torch.int32)
            # Data chỉ cần quan tâm đến tâm + class idx
            new_data.append([data[-1], data[0], data[1]])
        LP_type = "biendai"

        l_point = new_data[0]
        r_point = new_data[0]
        for cp in new_data:
            if cp[1] < l_point[1]:
                l_point = cp
            if cp[1] > r_point[1]:
                r_point = cp
        for ct in new_data:
            if l_point[1] != r_point[1]:
                if (check_point_linear(ct[1], ct[2], l_point[1], l_point[2], r_point[1], r_point[2]) == False):
                    LP_type = "bienvuong"
        # Xử lý biển vuông
        if LP_type == 'bienvuong':
            upper_data, lower_data = group_lines(new_data)

            sorted_upper_data = sorted(upper_data, key=lambda point: (point[1]))
            sorted_lower_data = sorted(lower_data, key=lambda point: (point[1]))
            for i in sorted_upper_data:
                lp_content += characters[i[0]]
            lp_content += ' '   
            for i in sorted_lower_data:
                lp_content += characters[i[0]]
        # Xử lý biển chữ nhật dài
        else:
            sorted_label_data = sorted(new_data, key=lambda point: (point[1]))
            
            for i in sorted_label_data:
                lp_content += characters[i[0]]

    return lp_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align License Plate')
    parser.add_argument('-img_path', '--img_path', type=str, help='lp img path for alignment', required=True)
    parser.add_argument('-model_path', '--model_path', type=str, help='model path for lp recognition', default='model/best_lp_recognition.pt')
    args = parser.parse_args()
    
    img_path = args.img_path
    img = cv2.imread(img_path)
    model = load_model(args.model_path)
    print(lp_recognition(img, model))
