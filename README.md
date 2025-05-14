# Face Recognition System

Hệ thống nhận diện khuôn mặt được xây dựng bằng Python, sử dụng OpenCV để phát hiện khuôn mặt, TensorFlow để huấn luyện mô hình CNN, và Tkinter cho giao diện người dùng. Dự án hỗ trợ thu thập dữ liệu khuôn mặt, huấn luyện mô hình, và nhận diện khuôn mặt trong thời gian thực, với khả năng gán nhãn "Unknown" cho những người chưa được huấn luyện.

## Mục lục
- [Mô tả](#mô-tả)
- [Yêu cầu](#yêu-cầu)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Lưu ý](#lưu-ý)
- [Giấy phép](#giấy-phép)

## Mô tả
Dự án này cung cấp một ứng dụng nhận diện khuôn mặt với các chức năng chính:
- **Thu thập dữ liệu**: Chụp ảnh khuôn mặt qua webcam và lưu vào thư mục `dataset`.
- **Huấn luyện mô hình**: Sử dụng CNN để học các đặc trưng khuôn mặt từ dữ liệu đã thu thập.
- **Nhận diện thời gian thực**: Phát hiện và nhận diện khuôn mặt từ webcam, hiển thị ID, tên, và độ tự tin (confidence score). Người chưa được huấn luyện sẽ được gán nhãn "Unknown".

## Yêu cầu
- **Hệ điều hành**: Windows, macOS, hoặc Linux.
- **Python**: Phiên bản 3.8 đến 3.11.
- **Webcam**: Thiết bị webcam hoạt động (thường là index 0).
- **Phần cứng**:
  - RAM tối thiểu 8GB (đề xuất 16GB để huấn luyện nhanh hơn).
  - CPU/GPU: GPU tùy chọn để tăng tốc huấn luyện với TensorFlow.
- **Thư viện Python**: Được liệt kê trong `requirements.txt`:
  - numpy==1.26.4
  - opencv-python==4.11.0.86
  - pandas==2.2.3
  - tensorflow==2.19.0
  - scikit-learn==1.6.1
  - Pillow==11.1.0

## Cài đặt
1. **Clone repository**:
   ```bash
   git clone https://github.com/duydk3012/face_recognition_system.git
   cd face_recognition_system
   ```

2. **Tạo môi trường ảo** (khuyến nghị):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Kiểm tra webcam**:
   - Đảm bảo webcam hoạt động. Nếu webcam không ở index 0, chỉnh sửa `self.camera_source` trong `face_recognition_system.py` (thử 0, 1, hoặc 2).

## Cách sử dụng
1. **Chạy ứng dụng**:
   ```bash
   python face_recognition_system.py
   ```

2. **Thu thập dữ liệu**:
   - Nhập **ID** (số) và **Name** vào giao diện.
   - Nhấn **Capture Data** để chụp 200 ảnh khuôn mặt.
   - Nhấn `q` hoặc `x` để dừng chụp.

3. **Huấn luyện mô hình**:
   - Nhấn **Train** để huấn luyện mô hình CNN.
   - Quá trình mất vài phút tùy thuộc vào số ảnh và phần cứng.

4. **Nhận diện khuôn mặt**:
   - Nhấn **Recognize** để bắt đầu nhận diện thời gian thực.
   - Kết quả hiển thị ID, tên, và độ tự tin (confidence score).
   - Người chưa được huấn luyện sẽ hiển thị nhãn "Unknown".
   - Nhấn `q` hoặc `x` để thoát.

## Cấu trúc dự án
```
face_recognition_system/
├── face_recognition_system.py  # Mã nguồn chính
├── requirements.txt            # Danh sách thư viện
├── .gitignore                  # Tệp bỏ qua các thư mục/tệp không cần thiết
├── README.md                   # Tài liệu này
```

**Lưu ý**: Các thư mục/tệp sau được tạo khi chạy chương trình nhưng không đẩy lên GitHub (do `.gitignore`):
- `dataset/`: Lưu ảnh khuôn mặt.
- `face_recognition_model.h5`: Mô hình đã huấn luyện.
- `face_info.csv`: Thông tin người dùng (ID và tên).

## Lưu ý
- **Webcam**: Nếu webcam không hoạt động, thử thay đổi `self.camera_source` trong `face_recognition_system.py` hoặc kiểm tra driver webcam.
- **Dữ liệu nhạy cảm**: Không chia sẻ `face_info.csv` hoặc `dataset/` công khai vì chúng có thể chứa thông tin cá nhân.
- **Hiệu suất**:
  - Để tăng độ chính xác, chụp ảnh ở nhiều góc độ và ánh sáng khác nhau.
  - Điều chỉnh `confidence_threshold` trong `recognize_face` (mặc định 0.7) nếu nhận diện nhầm hoặc từ chối nhầm.
- **Lỗi NumPy**: Code yêu cầu NumPy < 2.0. Nếu gặp lỗi, chạy:
  ```bash
  pip install numpy==1.26.4
  ```
- **Hiệu suất huấn luyện**: Sử dụng GPU (với CUDA/cuDNN) để tăng tốc nếu có.

## Giấy phép
[MIT License](LICENSE)

---
Dự án được phát triển bởi [duydk3012](https://github.com/duydk3012).