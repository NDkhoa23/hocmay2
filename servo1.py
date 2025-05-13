import os
import glob
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import RS304MD  # Thư viện điều khiển động cơ

# Khai báo các thông số
IMG_SIZE = 224  # Kích thước ảnh đầu vào cho mô hình
MODEL_PATH = "final_model.h5"  # Đường dẫn mô hình đã huấn luyện
TEST_PATH = "DataTest"  # Đường dẫn đến thư mục chứa ảnh kiểm tra
ID = 2  # ID của động cơ (servo)
THOIGIAN_DICHUYEN = 100  # Thời gian di chuyển của động cơ (ms)

# Load mô hình
model = load_model(MODEL_PATH)
rs = RS304MD.Rs()

def mocongCOM():
    """Mở kết nối COM và bật mô-men xoắn cho động cơ"""
    try:
        rs.open_port('COM3', 115200, 1)
        print('Kết nối COM3 thành công!')
        rs.torque_on(ID, 1)
        print('Enable Force Successfull!')
    except Exception as e:
        print(f"Lỗi kết nối COM: {e}")

def dongcongCOM():
    """Tắt mô-men xoắn và đóng kết nối COM"""
    try:
        rs.torque_on(ID, 0)
        rs.close_port()
        print('Hủy kết nối COM3 thành công!')
    except Exception as e:
        print(f"Lỗi khi đóng kết nối COM: {e}")

def chayrobot(gocdichuyen):
    """Điều khiển động cơ di chuyển đến góc yêu cầu"""
    print(f'Di chuyển đến góc {gocdichuyen} độ')
    try:
        rs.target_position(ID, gocdichuyen * 10, THOIGIAN_DICHUYEN)  # Di chuyển động cơ
        time.sleep(1)  # Đợi cho động cơ di chuyển xong
    except Exception as e:
        print(f"Lỗi khi điều khiển động cơ: {e}")

def load_and_predict_images_from_folder(folder):
    """Tải ảnh từ thư mục và dự đoán"""
    images = glob.glob(os.path.join(folder, "*.jpg"))  # Lấy tất cả ảnh từ thư mục
    for img_path in images:
        # Đọc và xử lý ảnh
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img / 255.0  # Chuẩn hóa ảnh
        img_input = np.expand_dims(img_norm, axis=0)  # Thêm chiều batch
        
        # Dự đoán
        pred = model.predict(img_input)[0][0]
        label = "ERROR" if pred >= 0.5 else "GOOD"
        print(f"{os.path.basename(img_path)} => {label} ({pred:.4f})")
        
        # Điều khiển động cơ dựa trên kết quả dự đoán
        chayrobot(90 if label == "ERROR" else -90)  # Di chuyển đến góc theo nhãn
        chayrobot(0)  # Quay về vị trí ban đầu (góc 0 độ)

def main():
    """Hàm chính để chạy thử nghiệm mô hình và điều khiển động cơ"""
    mocongCOM()  # Mở kết nối COM và bật mô-men xoắn
    
    # Dự đoán cho các ảnh trong thư mục 'good'
    print("Dự đoán cho các ảnh 'good':")
    load_and_predict_images_from_folder(os.path.join(TEST_PATH, 'good'))
    
    # Dự đoán cho các ảnh trong thư mục 'error'
    print("Dự đoán cho các ảnh 'error':")
    load_and_predict_images_from_folder(os.path.join(TEST_PATH, 'error'))
    
    dongcongCOM()  # Đóng kết nối COM sau khi hoàn tất

if __name__ == "__main__":
    print("Nguyễn Đăng Khoa 211604294")  # Thông tin tác giả
    main()  # Gọi hàm chính để thực thi
