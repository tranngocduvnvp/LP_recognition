import tqdm
import os
import glob
import matplotlib.pyplot as plt

def change_class_id(input_folder, output_folder):
    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lặp qua tất cả các file trong thư mục đầu vào
    for filename in tqdm.tqdm(os.listdir(input_folder)):
        if filename.endswith('.txt'):
            # Đường dẫn đầy đủ của file đầu vào
            input_file_path = os.path.join(input_folder, filename)

            # Đường dẫn đầy đủ của file đầu ra
            output_file_path = os.path.join(output_folder, filename)

            # Mở file đầu vào và đọc nội dung
            with open(input_file_path, 'r') as input_file:
                lines = input_file.readlines()

            # Sửa đổi ID của lớp và lưu vào file đầu ra
            modified_lines = []
            for line in lines:
                data = line.split()
                if data:  # Kiểm tra xem dòng có dữ liệu không
                    class_id = int(data[0])
                    if class_id in [0]:
                        data[0] = '0'
                    else:
                        data[0] = '1'
                    modified_lines.append(' '.join(data) + '\n')

            with open(output_file_path, 'w') as output_file:
                output_file.writelines(modified_lines)
                
                
if __name__ == "__main__":
    change_class_id("/home/bdi/Mammo_FDA/TensorRT/dataset/train/labels",\
                "/home/bdi/Mammo_FDA/TensorRT/dataset/train/labels")
    change_class_id("/home/bdi/Mammo_FDA/TensorRT/dataset/valid/labels",\
                "/home/bdi/Mammo_FDA/TensorRT/dataset/valid/labels")
    change_class_id("/home/bdi/Mammo_FDA/TensorRT/dataset/test/labels",\
                "/home/bdi/Mammo_FDA/TensorRT/dataset/test/labels")