import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from yolov5.detect import run
import os
import threading


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание объектов с YOLOv5")

        self.label = tk.Label(root, text="Загрузите изображение или видео для детекции")
        self.label.pack()

        self.upload_button = tk.Button(root, text="Загрузить файл", command=self.upload_file)
        self.upload_button.pack()

        self.webcam_button = tk.Button(root, text="Распознавание с web-камеры", command=self.run_webcam)
        self.webcam_button.pack()

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image and Video files", "*.jpg *.png *.jpeg *.mp4 *.avi")])
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.jpg', '.png', '.jpeg']:
            self.process_image(file_path)
        elif file_ext in ['.mp4', '.avi']:
            self.process_video(file_path)
        else:
            print("Неподдерживаемый тип файла")

    def run_webcam(self):
        # Функция для распознавания с веб-камеры
        def webcam_detection():
            run(
                weights='E:/PythonProjects/tl-detection/tl-detecion/yolov5/runs/train/exp12/weights/best.pt',  # Путь к вашей модели
                source='0',  # Веб-камера (индекс 0)
                imgsz=(640, 640),  # Размер изображения
                conf_thres=0.25,  # Порог уверенности
                iou_thres=0.45,  # Порог IoU для NMS
                max_det=1000,  # Максимальное количество объектов
                view_img=True,  # Отображать результаты в реальном времени
                nosave=True,  # Не сохранять изображения/видео
                project='runs/detect',  # Папка для результатов
                name='webcam',  # Имя подкаталога
                exist_ok=True  # Перезаписывать существующие результаты
            )

        # Запуск в отдельном потоке
        threading.Thread(target=webcam_detection).start()

    def process_image(self, file_path):
        run(
            weights='E:/PythonProjects/tl-detection/tl-detecion/yolov5/runs/train/exp12/weights/best.pt',
            source=file_path,
            save_txt=False,
            save_conf=False,
            nosave=False,
            project='runs/detect',
            name='exp',
            exist_ok=True
        )
        save_dir = os.path.join('runs', 'detect', 'exp')
        img_name = os.path.basename(file_path)
        detected_img_path = os.path.join(save_dir, img_name)
        self.display_image(detected_img_path)

    def display_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

    def process_video(self, file_path):
        run(
            weights='E:/PythonProjects/tl-detection/tl-detecion/yolov5/runs/train/exp12/weights/best.pt',
            source=file_path,
            save_txt=False,
            save_conf=False,
            nosave=False,
            project='runs/detect',
            name='exp',
            exist_ok=True
        )
        save_dir = os.path.join('runs', 'detect', 'exp')
        video_name = os.path.basename(file_path)
        detected_video_path = os.path.join(save_dir, video_name)

        def play_video():
            cap = cv2.VideoCapture(detected_video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Detected Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=play_video).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()