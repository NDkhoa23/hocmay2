import os
import glob
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import RS304MD  # Thư viện điều khiển động cơ

# Khai báo các thông số
IMG_SIZE = 224  
MODEL_PATH = "best_restnet_model.h5"  
TEST_PATH = "DataTest"  
ID = 2  # ID của động cơ
THOIGIAN_DICHUYEN = 100  # Thời gian di chuyển của động cơ

# Load mô hình
model = load_model(MODEL_PATH)
rs = RS304MD.Rs()

def mocongCOM():
    rs.open_port('COM3', 115200, 1)
    print('Kết nối COM3 thành công!')
    rs.torque_on(ID, 1)
    print('Enable Force Successfull!')

def dongcongCOM():
    rs.torque_on(ID, 0)
    rs.close_port()
    print('Hủy kết nối COM3 thành công!')

def chayrobot(gocdichuyen):
    print(f'Di chuyển đến góc {gocdichuyen} độ')
    rs.target_position(ID, gocdichuyen * 10, THOIGIAN_DICHUYEN)
    time.sleep(1)

def main():
    mocongCOM()
    
    test_images = glob.glob(os.path.join(TEST_PATH, "*"))
    
    for img_path in test_images:
        # Đọc và xử lý ảnh
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img / 255.0
        img_input = np.expand_dims(img_norm, axis=0)
        
        # Dự đoán
        pred = model.predict(img_input)[0][0]
        label = "ERROR" if pred >= 0.5 else "GOOD"
        print(f"{os.path.basename(img_path)} => {label} ({pred:.4f})")
        
        # Điều khiển động cơ
        chayrobot(90 if label == "ERROR" else -90)
        chayrobot(0)  # Quay về vị trí ban đầu
    
    dongcongCOM()

if __name__ == "__main__":
    print("Nguyễn Đăng Khoa 211604294")
    main()
