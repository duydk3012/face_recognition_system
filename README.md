Face Recognition System
Hệ thống nhận diện khuôn mặt được xây dựng bằng Python, sử dụng OpenCV để phát hiện khuôn mặt, TensorFlow để huấn luyện mô hình CNN, và Tkinter cho giao diện người dùng. Dự án hỗ trợ thu thập dữ liệu khuôn mặt, huấn luyện mô hình, và nhận diện khuôn mặt trong thời gian thực, với khả năng gán nhãn "Unknown" cho những người chưa được huấn luyện.
Mục lục

Mô tả
Yêu cầu
Cài đặt
Cách sử dụng
Cấu trúc dự án
Lưu ý
Giấy phép

Mô tả
Dự án này cung cấp một ứng dụng nhận diện khuôn mặt với các chức năng chính:

Thu thập dữ liệu: Chụp ảnh khuôn mặt qua webcam và lưu vào thư mục dataset.
Huấn luyện mô hình: Sử dụng CNN để học các đặc trưng khuôn mặt từ dữ liệu đã thu thập.
Nhận diện thời gian thực: Phát hiện và nhận diện khuôn mặt từ webcam, hiển thị ID, tên, và độ tự tin (confidence score). Người chưa được huấn luyện sẽ được gán nhãn "Unknown".

Yêu cầu

Hệ điều hành: Windows, macOS, hoặc Linux.
Python: Phiên bản 3.8 đến 3.11.
Webcam: Thiết bị webcam hoạt động (thường là index 0).
Phần cứng:
RAM tối thiểu 8GB (đề xuất 16GB để huấn luyện nhanh hơn).
CPU/GPU: GPU tùy chọn để tăng tốc huấn luyện với TensorFlow.


Thư viện Python: Được liệt kê trong requirements.txt:
numpy==1.26.4
opencv-python==4.10.0
pandas==2.2.3
tensorflow==2.17.0
scikit-learn==1.5.2
pillow==10.4.0



Cài đặt

Clone repository:
git clone https://github.com/duydk3012/face_recognition_system.git
cd face_recognition_system


Tạo môi trường ảo (khuyến nghị):
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Cài đặt thư viện:
pip install -r requirements.txt


Kiểm tra webcam:

Đảm bảo webcam hoạt động. Nếu webcam không ở index 0, chỉnh sửa self.camera_source trong face_recognition_system.py (thử 0, 1, hoặc 2).



Cách sử dụng

Chạy ứng dụng:
python face_recognition_system.py


Thu thập dữ liệu:

Nhập ID (số) và Name vào giao diện.
Nhấn Capture Data để chụp 200 ảnh khuôn mặt.
Nhấn q hoặc x để dừng chụp.


Huấn luyện mô hình:

Nhấn Train để huấn luyện mô hình CNN.
Quá trình mất vài phút tùy thuộc vào số ảnh và phần cứng.


Nhận diện khuôn mặt:

Nhấn Recognize để bắt đầu nhận diện thời gian thực.
Kết quả hiển thị ID, tên, và độ tự tin (confidence score).
Người chưa được huấn luyện sẽ hiển thị nhãn "Unknown".
Nhấn q hoặc x để thoát.



Cấu trúc dự án
face_recognition_system/
├── face_recognition_system.py  # Mã nguồn chính
├── requirements.txt            # Danh sách thư viện
├── .gitignore                  # Tệp bỏ qua các thư mục/tệp không cần thiết
├── README.md                   # Tài liệu này

Lưu ý: Các thư mục/tệp sau được tạo khi chạy chương trình nhưng không đẩy lên GitHub (do .gitignore):

dataset/: Lưu ảnh khuôn mặt.
face_recognition_model.h5: Mô hình đã huấn luyện.
face_info.csv: Thông tin người dùng (ID và tên).

Lưu ý

Webcam: Nếu webcam không hoạt động, thử thay đổi self.camera_source trong face_recognition_system.py hoặc kiểm tra driver webcam.
Dữ liệu nhạy cảm: Không chia sẻ face_info.csv hoặc dataset/ công khai vì chúng có thể chứa thông tin cá nhân.
Hiệu suất:
Để tăng độ chính xác, chụp ảnh ở nhiều góc độ và ánh sáng khác nhau.
Điều chỉnh confidence_threshold trong recognize_face (mặc định 0.7) nếu nhận diện nhầm hoặc từ chối nhầm.


Lỗi NumPy: Code yêu cầu NumPy < 2.0. Nếu gặp lỗi, chạy:pip install numpy==1.26.4


Hiệu suất huấn luyện: Sử dụng GPU (với CUDA/cuDNN) để tăng tốc nếu có.

Giấy phép
MIT License

Dự án được phát triển bởi duydk3012.
