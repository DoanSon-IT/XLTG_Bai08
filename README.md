## Sinh viên thực hiện
- **Họ và tên**: Đoàn Văn Sơn  
- **Mã sinh viên**: 20211512  
- **Lớp**: DCCNTT12.10.6  

---

# Thực hành: Phân loại gà và vịt

### Mô tả bài toán  
Bài toán phân loại chó và mèo sử dụng các mô hình **ANN**, **CNN**, và **R-CNN** được chuyển đổi thành<br>
bài toán phân loại **gà** và **vịt** do trên lớp chưa tìm thấy dataset của dog & cat.

- **Dữ liệu đầu vào**:  
  - Ảnh và nhãn được chia thành hai tập:  
    - **Train**: ~100 ảnh mỗi loại  
    - **Validation**: ~100 ảnh mỗi loại  

- **Mô hình sử dụng**:  
  Sử dụng ba mô hình **ANN**, **CNN**, và **R-CNN** với tham số **batch_size = 32**.  
  - Việc lựa chọn **batch_size = 32** dựa trên phân tích biểu đồ trong file: `lua_chon_batch_size.png`.  

### Quy trình dự đoán  
1. **Ảnh test**:  
   - Ảnh được đặt trong thư mục `data/test`.  

2. **Kết quả dự đoán**:  
   - Kết quả được lưu trong thư mục `data/output`.  
   - Ảnh test sau khi dự đoán sẽ được gán nhãn **gà** hoặc **vịt**, hiển thị trực tiếp trên từng ảnh.  
