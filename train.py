import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Đường dẫn tới dữ liệu huấn luyện và kiểm tra
train_dir = 'DataTranning'
val_dir = 'DataTest'

# Kích thước ảnh đầu vào cho MobileNetV2
IMG_SIZE = 224  # 224x224 pixel là kích thước yêu cầu cho MobileNetV2

# Các thông số huấn luyện
BATCH_SIZE = 32
EPOCHS = 50

# Tạo ImageDataGenerator với tăng cường dữ liệu cho bộ huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Tăng phạm vi xoay
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.5, 1.5]  # Điều chỉnh độ sáng
)

# Tạo ImageDataGenerator cho bộ kiểm tra (chỉ scale ảnh)
test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generators cho dữ liệu huấn luyện và kiểm tra
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 2 lớp: 'good' và 'error'
)

test_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Sử dụng MobileNetV2 pre-trained với trọng số 'imagenet'
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')

# Đóng băng các lớp của base model
base_model.trainable = False

# Xây dựng mô hình với MobileNetV2
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Lớp đầu ra cho 2 lớp phân loại
])

# Compile mô hình
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Định nghĩa các callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Lưu mô hình cuối cùng
model.save('final_model.h5')

# Đánh giá mô hình trên bộ kiểm tra
loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Nếu muốn fine-tune sau một số epoch, mở khóa một số lớp của base model
base_model.trainable = True
fine_tune_at = 100  # Fine-tune từ lớp thứ 100 trở đi

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile lại mô hình với learning rate nhỏ hơn
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Tiếp tục huấn luyện mô hình sau khi fine-tune
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Đánh giá lại sau fine-tuning
loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy after fine-tuning: {accuracy * 100:.2f}%")
