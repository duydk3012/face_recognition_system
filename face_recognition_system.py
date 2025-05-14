import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import os
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Kiểm tra NumPy tương thích
try:
    import numpy
    if numpy.__version__.startswith('2'):
        messagebox.showerror("Error", "NumPy 2.x không tương thích. Vui lòng cài đặt NumPy 1.26.4 bằng: pip install numpy==1.26.4")
        exit()
except ImportError as e:
    messagebox.showerror("Error", f"Lỗi khi import thư viện: {str(e)}. Vui lòng cài đặt các thư viện cần thiết.")
    exit()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry('400x300')
        self.root.configure(bg='#263D42')

        # Thông số
        self.DATASET_DIR = "dataset"
        self.MODEL_PATH = "face_recognition_model.h5"
        self.INFO_CSV = "face_info.csv"
        self.IMG_SIZE = (128, 128)
        self.BATCH_SIZE = 32

        # Khởi tạo thư mục dataset
        if not os.path.exists(self.DATASET_DIR):
            os.makedirs(self.DATASET_DIR)

        # Tải Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            messagebox.showerror("Error", "Không thể tải bộ phân loại Haar Cascade!")
            exit()

        # Chỉ số camera (có thể thay đổi nếu webcam không phải index 0)
        self.camera_source = 1  # Đổi thành 0 (thường là webcam mặc định)
        self.setup_gui()

    def setup_gui(self):
        # Tiêu đề
        ttk.Label(self.root, text="Face Recognition System", background="grey", foreground="white",
                  font=("Arial", 14)).place(x=100, y=20)

        # Nhãn và trường nhập
        ttk.Label(self.root, text="ID:", background="#263D42", foreground="white").place(x=50, y=80)
        ttk.Label(self.root, text="Name:", background="#263D42", foreground="white").place(x=50, y=120)

        self.id_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.edit_id = ttk.Entry(self.root, textvariable=self.id_var, width=40)
        self.edit_id.place(x=100, y=80)
        self.edit_id.focus()
        self.edit_name = ttk.Entry(self.root, textvariable=self.name_var, width=40)
        self.edit_name.place(x=100, y=120)

        # Nút chức năng
        ttk.Button(self.root, text="Capture Data", command=self.capture_data).place(x=50, y=180)
        ttk.Button(self.root, text="Train", command=self.train_data).place(x=170, y=180)
        ttk.Button(self.root, text="Recognize", command=self.recognize_face).place(x=290, y=180)

    def capture_data(self):
        try:
            user_id = self.id_var.get().strip()
            user_name = self.name_var.get().strip()
            if not user_id or not user_name:
                raise ValueError("ID và Name không được để trống!")
            if not user_id.isdigit():
                raise ValueError("ID phải là số!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        # Kiểm tra và ghi tiêu đề CSV
        if not os.path.exists(self.INFO_CSV):
            with open(self.INFO_CSV, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['id', 'name'])

        # Ghi thông tin người dùng vào CSV
        with open(self.INFO_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([user_id, user_name])

        # Kiểm tra webcam
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Không thể truy cập camera! Kiểm tra kết nối hoặc chỉ số camera.")
            return

        sample_num = 0
        max_images = 200  # Tăng số ảnh chụp lên 200 để cải thiện dữ liệu
        font = cv2.FONT_HERSHEY_SIMPLEX

        while sample_num < max_images:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Không thể đọc từ webcam!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                continue  # Bỏ qua nếu không phát hiện khuôn mặt

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, self.IMG_SIZE)
                img_path = os.path.join(self.DATASET_DIR, f"{user_id}_{sample_num}.jpg")
                cv2.imwrite(img_path, face_img)
                sample_num += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            counter_text = f"{sample_num}/{max_images}"
            text_size = cv2.getTextSize(counter_text, font, 1.2, 2)[0]
            cv2.rectangle(frame, (10, 10), (10 + text_size[0], 10 + text_size[1] + 10), (0, 0, 0), -1)
            cv2.putText(frame, counter_text, (10, 30), font, 1.2, (255, 255, 255), 2)

            cv2.imshow('Capture', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('x'):
                break
            if cv2.getWindowProperty('Capture', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", f"Đã chụp {sample_num} ảnh cho {user_name} (ID: {user_id})!")
        self.edit_id.delete(0, "end")
        self.edit_name.delete(0, "end")

    def build_model(self, num_classes):
        model = Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same'),  # Thêm tầng Conv2D
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),  # Tăng số nơ-ron
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_data(self):
        if not os.path.exists(self.INFO_CSV):
            messagebox.showerror("Error", "Không tìm thấy dữ liệu người dùng! Vui lòng chụp dữ liệu trước.")
            return

        try:
            df = pd.read_csv(self.INFO_CSV)
            if df.empty:
                messagebox.showerror("Error", "Không có người dùng nào trong face_info.csv!")
                return
            user_ids = df['id'].unique()
            num_classes = len(user_ids)
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi đọc face_info.csv: {str(e)}")
            return

        # Progressbar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=200, mode="determinate")
        self.progress.place(x=100, y=250)
        self.progress_label = ttk.Label(self.root, text="0%")
        self.progress_label.place(x=310, y=250)

        # Chuẩn bị dữ liệu
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,  # Tăng xoay
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],  # Thay đổi độ sáng
            horizontal_flip=True
        )

        images = []
        labels = []
        for user_id in user_ids:
            for img_name in os.listdir(self.DATASET_DIR):
                if img_name.startswith(f"{user_id}_"):
                    img_path = os.path.join(self.DATASET_DIR, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, self.IMG_SIZE)
                    images.append(img)
                    labels.append(user_id)

        if not images:
            self.progress.destroy()
            self.progress_label.destroy()
            messagebox.showerror("Error", "Không tìm thấy ảnh trong dataset!")
            return

        images = np.array(images)
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.utils import to_categorical
        label_encoder = LabelEncoder()
        integer_labels = label_encoder.fit_transform(labels)
        one_hot_labels = to_categorical(integer_labels, num_classes=num_classes)

        # Cập nhật progressbar
        self.progress['maximum'] = 20  # Tăng số epoch lên 20
        self.progress['value'] = 0

        # Huấn luyện mô hình
        try:
            model = self.build_model(num_classes)
            from tensorflow.keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            for epoch in range(20):
                model.fit(
                    datagen.flow(images, one_hot_labels, batch_size=self.BATCH_SIZE),
                    epochs=1,
                    callbacks=[early_stopping],
                    verbose=0
                )
                percent = int((epoch + 1) / 20 * 100)
                self.progress['value'] = epoch + 1
                self.progress_label.config(text=f"{percent}%")
                self.root.update_idletasks()
                time.sleep(0.1)

            model.save(self.MODEL_PATH)
            self.progress['value'] = 20
            self.progress_label.config(text="100%")
            self.root.update_idletasks()
            time.sleep(0.5)
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi huấn luyện: {str(e)}")
            return
        finally:
            self.progress.destroy()
            self.progress_label.destroy()

        messagebox.showinfo("Success", "Huấn luyện hoàn tất thành công!")

    def recognize_face(self):
        if not os.path.exists(self.MODEL_PATH):
            messagebox.showerror("Error", "Mô hình chưa được huấn luyện!")
            return

        try:
            model = tf.keras.models.load_model(self.MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi tải mô hình: {str(e)}")
            return

        try:
            df = pd.read_csv(self.INFO_CSV)
            if df.empty:
                messagebox.showerror("Error", "Không có dữ liệu người dùng!")
                return
            id_to_name = dict(zip(df['id'], df['name']))
            user_ids = df['id'].unique()
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi đọc face_info.csv: {str(e)}")
            return

        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Không thể truy cập camera!")
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        confidence_threshold = 0.7  # Ngưỡng xác suất

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Không thể đọc từ webcam!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, self.IMG_SIZE)
                face_img = face_img / 255.0
                face_img = np.expand_dims(face_img, axis=0)
                predictions = model.predict(face_img, verbose=0)
                max_confidence = np.max(predictions)
                if max_confidence < confidence_threshold:
                    predicted_name = "Unknown"
                    predicted_id = -1
                else:
                    predicted_id = user_ids[np.argmax(predictions)]
                    predicted_name = id_to_name.get(predicted_id, "Unknown")
                label = f"ID: {predicted_id}, Name: {predicted_name}, Conf: {max_confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), font, 0.9, (0, 255, 0), 2)

            cv2.imshow('Recognition', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('x'):
                break
            if cv2.getWindowProperty('Recognition', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    win = tk.Tk()
    app = FaceRecognitionApp(win)
    win.mainloop()