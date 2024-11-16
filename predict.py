import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
import numpy as np
import cv2

# Danh sách mô hình
model_paths = {
    'ANN': 'models/ann_model.h5',
    'CNN': 'models/cnn_model.h5',
    'RCNN': 'models/rcnn_model.h5'
}

# Thư mục đầu vào và đầu ra
input_folder = './test'  # Thư mục chứa ảnh cần dự đoán
output_folder = './output'
os.makedirs(output_folder, exist_ok=True)

# Hàm dự đoán và vẽ nhãn
def predict_and_draw(image_path, model, model_name):
    # Đọc ảnh
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_normalized = img_array / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Dự đoán
    prediction = model.predict(img_expanded)[0][0]
    label = "Duck" if prediction > 0.5 else "Chicken"

    # Đọc ảnh gốc
    original_img = cv2.imread(image_path)
    h, w, _ = original_img.shape

    # Vẽ nhãn lên ảnh
    cv2.putText(original_img, f"{model_name}: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Lưu ảnh vào thư mục output
    output_path = os.path.join(output_folder, f"{model_name}_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, original_img)
    print(f"Saved output to {output_path}")

# Duyệt qua các mô hình và thực hiện dự đoán
for model_name, model_path in model_paths.items():
    print(f"Loading model: {model_name}")
    model = load_model(model_path)
    
    # Dự đoán cho mỗi ảnh trong thư mục test
    for img_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, img_file)
        predict_and_draw(image_path, model, model_name)
