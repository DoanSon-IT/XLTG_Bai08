import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau # type: ignore
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore

# Tạo thư mục lưu mô hình nếu chưa có
os.makedirs('models', exist_ok=True)

# Đường dẫn dữ liệu
train_dir = 'data/train'
val_dir = 'data/validation'

# Kích thước ảnh
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Tạo mô hình RCNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kỹ thuật Learning Rate Warmup
def lr_warmup(epoch, lr):
    if epoch < 5:
        return lr * (epoch + 1) / 5  # Tăng dần trong 5 epoch đầu
    return lr

lr_scheduler = LearningRateScheduler(lr_warmup)

# Kỹ thuật Learning Rate Decay
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.0001)

# Kỹ thuật Label Smoothing
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

# Huấn luyện mô hình
print("Training RCNN model with improved techniques...")
model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[lr_scheduler, lr_reduction])

# Lưu mô hình
model.save('models/rcnn_model.h5')
print("RCNN model saved to models/rcnn_model.h5")
