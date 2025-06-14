import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox, Label
from PIL import Image, ImageTk

# Tắt thông báo của TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model đã được huấn luyện
model = load_model("final_emotion_model_gray.h5")

# Tải class_names từ file JSON hoặc định nghĩa trực tiếp
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Tải Cascade Classifier cho nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Tải emoji cho từng cảm xúc
emotion_emojis = {}

for emotion in class_names:
    emoji_path = next(
        (path for path in [f"{emotion}.png", f"{emotion}.jpg"] if os.path.exists(path)),
        None,
    )
    if emoji_path:
        emotion_emojis[emotion] = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    else:
        emotion_emojis[emotion] = None


# Hàm nhận diện cảm xúc qua Webcam
def analyze_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi sang grayscale cho nhận diện khuôn mặt
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            # Vẽ khung nhận diện
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Trích xuất vùng khuôn mặt từ ảnh grayscale
            roi_gray = gray_frame[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            # Chuẩn hóa ảnh
            roi_gray = roi_gray.astype("float32") / 255.0
            roi_gray = np.expand_dims(
                roi_gray, axis=(0, -1)
            )  # Thêm batch dimension và channel dimension

            # Dự đoán cảm xúc
            predictions = model.predict(roi_gray)
            emotion_index = np.argmax(predictions[0])
            emotion_label = class_names[emotion_index]

            # Hiển thị tên cảm xúc
            cv2.putText(
                frame,
                emotion_label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Hiển thị emoji tương ứng
            emoji = emotion_emojis.get(emotion_label)
            if emoji is not None:
                emoji_size = (w // 5, h // 5)
                emoji = cv2.resize(emoji, emoji_size)
                x_offset = x + w - emoji_size[0]
                y_offset = y - emoji_size[1] - 10
                y_offset = max(0, y_offset)

                if emoji.shape[2] == 4:
                    alpha = emoji[:, :, 3] / 255.0
                    for c in range(3):
                        frame[
                            y_offset : y_offset + emoji_size[1],
                            x_offset : x_offset + emoji_size[0],
                            c,
                        ] = (
                            alpha * emoji[:, :, c]
                            + (1 - alpha)
                            * frame[
                                y_offset : y_offset + emoji_size[1],
                                x_offset : x_offset + emoji_size[0],
                                c,
                            ]
                        )
                else:
                    frame[
                        y_offset : y_offset + emoji_size[1],
                        x_offset : x_offset + emoji_size[0],
                    ] = emoji[:, :, :3]

        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Hàm nhận diện cảm xúc từ ảnh
def analyze_from_image():
    image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if not image_path:
        return

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        messagebox.showinfo("Kết quả", "Không tìm thấy khuôn mặt nào.")
        return

    # Hiển thị hình ảnh được chọn
    img = Image.open(image_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    x, y, w, h = faces[0]
    # Trích xuất vùng khuôn mặt từ ảnh grayscale
    roi_gray = gray_image[y : y + h, x : x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    # Chuẩn hóa ảnh
    roi_gray = roi_gray.astype("float32") / 255.0
    roi_gray = np.expand_dims(
        roi_gray, axis=(0, -1)
    )  # Thêm batch dimension và channel dimension

    # Dự đoán cảm xúc
    predictions = model.predict(roi_gray)
    emotion_index = np.argmax(predictions[0])
    emotion_label = class_names[emotion_index]

    result_text = f"Cảm xúc: {emotion_label}"
    emoji_image = emotion_emojis.get(emotion_label)

    # Hiển thị kết quả
    result_label.config(text=result_text)

    # Hiển thị emoji nếu có
    if emoji_image is not None:
        emoji_img = cv2.cvtColor(emoji_image, cv2.COLOR_BGRA2RGBA)
        emoji_img = Image.fromarray(emoji_img)
        emoji_img.thumbnail((64, 64))
        emoji_img = ImageTk.PhotoImage(emoji_img)
        emoji_label.config(image=emoji_img)
        emoji_label.image = emoji_img
    else:
        emoji_label.config(image="")


# Giao diện Tkinter
root = tk.Tk()
root.title("Emotion Recognition App")
root.geometry("600x400")

btn_webcam = tk.Button(
    root, text="Nhận diện qua Webcam", width=25, command=analyze_from_webcam
)
btn_image = tk.Button(
    root, text="Nhận diện qua Ảnh", width=25, command=analyze_from_image
)

btn_webcam.pack(pady=10)
btn_image.pack(pady=10)

image_label = Label(root)
image_label.pack()

emoji_label = Label(root)
emoji_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
