import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget, QVBoxLayout, QLabel
import time
import os
from ultralytics import YOLO
from collections import Counter
import difflib
import easyocr
import re


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # RTSP URL kamera
        url = "rtsp://itdev:itdev@2025@192.168.160.54:554/cam/realmonitor?channel=1&subtype=0"
        url2 = "rtsp://itdev:itdev@2025@192.168.160.54:554/cam/realmonitor?channel=2&subtype=0"
        url3 = "rtsp://itdev:itdev@2025@192.168.160.54:554/cam/realmonitor?channel=3&subtype=0"

        self.cap1 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(url)
        self.cap3 = cv2.VideoCapture(url2)
        self.cap4 = cv2.VideoCapture(url3)

        # if not self.cap1.isOpened() or not self.cap2.isOpened():
        #     print("Error: Tidak bisa membuka kamera.")
        #     sys.exit()
        # Set the central widget.
        self.setMinimumSize(800, 600)
        self.showMaximized()
        self.setWindowTitle("IP Camera System Container Damage")
        # self.setGeometry(100, 100, 1200, 600)

        # self.layout = QVBoxLayout()
        # self.label1 = QLabel(self)
        # self.label2 = QLabel(self)
        # self.label3 = QLabel(self)
        # self.label4 = QLabel(self)
        # self.layout.addWidget(self.label1)
        # self.layout.addWidget(self.label2)
        # self.layout.addWidget(self.label3)
        # self.layout.addWidget(self.label4)
        # self.setLayout(self.layout)

        # Ganti layout menjadi QGridLayout
        self.layout = QGridLayout()
        self.label1 = QLabel(self)
        self.label2 = QLabel(self)
        self.label3 = QLabel(self)
        self.label4 = QLabel(self)

        # Tambahkan label ke dalam grid layout
        self.layout.addWidget(self.label1, 0, 0)
        self.layout.addWidget(self.label2, 0, 1)
        self.layout.addWidget(self.label3, 1, 0)
        self.layout.addWidget(self.label4, 1, 1)

        self.setLayout(self.layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.previous_positions = {}

        # Folder untuk menyimpan hasil capture
        self.save_dir = "captured"
        os.makedirs(self.save_dir, exist_ok=True)

        # Load dan siapkan background
        # self.background = cv2.imread("background2.jpg")
        # if self.background is not None:
        #     ret_test, test_frame = self.cap1.read()
        #     if ret_test:
        #         test_height, test_width = test_frame.shape[:2]
        #         self.background = cv2.resize(
        #             self.background, (test_width, test_height))
        #         self.background = cv2.cvtColor(
        #             self.background, cv2.COLOR_BGR2GRAY)
        #         self.background = cv2.GaussianBlur(
        #             self.background, (21, 21), 0)
        #     else:
        #         print("Tidak dapat membaca frame kamera untuk resize background.")
        #         sys.exit()
        # else:
        #     print("Background image not found!")
        #     sys.exit()

        self.prev_gray = None  # Untuk frame sebelumnya

        self.cooldown_active = False
        self.cooldown_start_time = 0
        self.last_draw_time = 0  # Timestamp terakhir bounding box dicetak

        # Load YOLOv8 model
        weight_path = os.path.abspath("Weights/yolov8.pt")
        self.model = YOLO(weight_path)
        self.class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']

    def generate_black_qimage(self, width, height):
        black_frame = np.zeros(
            (height, width, 3), dtype=np.uint8)  # Hitam penuh
        rgb_black = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb_black.data, width, height, width * 3, QImage.Format_RGB888)

    def ocr_image_to_text(self, image_path):

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Gagal membaca gambar dari path: {image_path}")
            return ""

        # --- STEP 1: Zoom image (digital zoom) ---
        zoom_factor = 2.0  # Ubah sesuai kebutuhan
        height, width = img.shape[:2]
        img_zoomed = cv2.resize(
            img, (int(width * zoom_factor), int(height * zoom_factor)))

        # Inisialisasi
        valid_words = ['22G1', '22G0', '45R1', '22T1']
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(img_zoomed)

        if not results:
            print("Tidak ada teks yang terdeteksi dalam gambar.")
            return ""

        corrected_texts = []

        for bbox, raw_text, score in results:
            text_upper = raw_text.upper()
            text_replaced = text_upper.replace('6', 'G')

            # Cari kemiripan terbaik dengan valid_words
            match = (difflib.get_close_matches(text_replaced, valid_words, n=1, cutoff=0.8) or
                     difflib.get_close_matches(text_upper, valid_words, n=1, cutoff=0.8))

            final_text = match[0] if match else text_upper
            cleaned_text = re.sub(r'[\[\]]', '', final_text)

            corrected_texts.append((bbox, cleaned_text, score))

        # Tampilkan hasil
        for _, text, _ in corrected_texts:
            print(text)

    def detect_damage_yolo(self, image_path):
        results = self.model(image_path)[0]
        # Baca ulang gambar
        img = cv2.imread(image_path)
        damage_counter = Counter()
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = self.class_labels[cls_id] if cls_id < len(
                self.class_labels) else "Unknown"
            conf = float(box.conf[0])

            # Hitung jumlah tiap kategori
            damage_counter[label] += 1

            # Gambar bounding box dan label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Tambahkan info jumlah di gambar
        y_offset = 20
        cv2.putText(img, f"Total Damage: {sum(damage_counter.values())}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        for label, count in damage_counter.items():
            cv2.putText(img, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

        # Simpan hasil dengan anotasi
        annotated_path = image_path.replace(".jpg", "_detected.jpg")
        cv2.imwrite(annotated_path, img)
        print(f"Hasil deteksi disimpan: {annotated_path}")
        print(f"Total Damage: {sum(damage_counter.values())}")
        print("Per kategori:")
        for label, count in damage_counter.items():
            print(f" - {label}: {count}")

    def update_frame(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        ret3, frame3 = self.cap3.read()
        ret4, frame4 = self.cap4.read()

        # background = self.background

        if ret1:
            clean_frame1 = frame1.copy()
            height1, width1, _ = frame1.shape
            # center_x = width1 // 2
            center_x = width1 - 100
            # center_x = int(width1 * 0.75)  # Dinamis berdasarkan lebar

            # Garis tengah hijau
            cv2.line(frame1, (center_x, 0),
                     (center_x, height1), (0, 255, 0), 2)

            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.prev_gray is None:
                self.prev_gray = gray
                return

            # diff = cv2.absdiff(self.background, gray)
            # Deteksi perubahan dari frame sebelumnya
            diff = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # current_positions = {}
            current_time = time.time()

            # Periksa apakah sedang dalam cooldown
            if self.cooldown_active:
                if current_time - self.cooldown_start_time >= 10:
                    self.cooldown_active = False  # Akhiri cooldown
                else:
                    # Tampilkan frame tanpa proses deteksi garis
                    rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    qimg1 = QImage(rgb1.data, width1, height1,
                                   width1 * 3, QImage.Format_RGB888)
                    self.label1.setPixmap(QPixmap.fromImage(qimg1))
                    return  # Lewati frame ini selama cooldown

            for contour in cnts:
                if cv2.contourArea(contour) < 10000:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cx = x + w // 2

                if x <= center_x <= x + w:
                    print("Bounding Box Touched the Line!")
                    self.cooldown_active = True
                    self.cooldown_start_time = current_time
                    # Simpan gambar dari kamera 1 dan 2
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename1 = os.path.join(
                        self.save_dir, f"cam1_{timestamp}.jpg")
                    filename2 = os.path.join(
                        self.save_dir, f"cam2_{timestamp}.jpg")
                    filename3 = os.path.join(
                        self.save_dir, f"cam3_{timestamp}.jpg")
                    filename4 = os.path.join(
                        self.save_dir, f"cam4_{timestamp}.jpg")
                    cv2.imwrite(filename1, clean_frame1)

                    if ret2:
                        cv2.imwrite(filename2, frame2)
                    if ret3:
                        cv2.imwrite(filename3, frame3)
                    if ret4:
                        cv2.imwrite(filename4, frame4)
                    print(
                        f"Gambar disimpan ke: {filename1} dan {filename2 if ret2 else '[kamera 2 gagal]'}")

                    self.detect_damage_yolo(filename2)
                    self.ocr_image_to_text(filename2)
                    break

                # Gambar bounding box
                cv2.rectangle(frame1, (x, y),
                              (x + w, y + h), (0, 255, 0), 3)

                # # Gambar bounding box
                # cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)

            rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            qimg1 = QImage(rgb1.data, width1, height1,
                           width1 * 3, QImage.Format_RGB888)
            self.label1.setPixmap(QPixmap.fromImage(qimg1))
        else:
            # black_img1 = self.generate_black_qimage(640, 480)
            black_img1 = self.generate_black_qimage(640, 480)
            self.label1.setPixmap(QPixmap.fromImage(black_img1))

        if ret2:
            height2, width2, _ = frame2.shape
            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            qimg2 = QImage(rgb2.data, width2, height2,
                           width2 * 3, QImage.Format_RGB888)
            self.label2.setPixmap(QPixmap.fromImage(qimg2))
        else:
            black_img = self.generate_black_qimage(640, 4800)
            self.label2.setPixmap(QPixmap.fromImage(black_img))

        if ret3:
            height3, width3, _ = frame3.shape
            rgb3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            qimg3 = QImage(rgb3.data, width3, height3,
                           width3 * 3, QImage.Format_RGB888)
            self.label3.setPixmap(QPixmap.fromImage(qimg3))
        else:
            black_img = self.generate_black_qimage(640, 480)
            self.label3.setPixmap(QPixmap.fromImage(black_img))

        if ret4:
            height4, width4, _ = frame4.shape
            rgb4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
            qimg4 = QImage(rgb4.data, width4, height4,
                           width4 * 3, QImage.Format_RGB888)
            self.label4.setPixmap(QPixmap.fromImage(qimg4))
        else:
            black_img = self.generate_black_qimage(640, 480)
            self.label4.setPixmap(QPixmap.fromImage(black_img))

    def closeEvent(self, event):
        self.cap1.release()
        self.cap2.release()
        self.cap3.release()
        self.cap4.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
