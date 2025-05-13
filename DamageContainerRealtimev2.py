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

        # Ganti layout menjadi QGridLayout
        self.layout = QGridLayout()

        # Menambahkan nama kamera di atas label untuk setiap kamera
        self.label1_name = QLabel("Kamera 1", self)
        self.label2_name = QLabel("Kamera 2", self)
        self.label3_name = QLabel("Kamera 3", self)
        self.label4_name = QLabel("Kamera 4", self)

        self.label1 = QLabel(self)
        self.label2 = QLabel(self)
        self.label3 = QLabel(self)
        self.label4 = QLabel(self)

        # Menambahkan nama kamera dan frame ke dalam layout
        self.layout.addWidget(self.label1, 0, 0)
        self.layout.addWidget(self.label2, 0, 1)
        self.layout.addWidget(self.label3, 1, 0)
        self.layout.addWidget(self.label4, 1, 1)

        self.label1.setScaledContents(True)
        self.label2.setScaledContents(True)
        self.label3.setScaledContents(True)
        self.label4.setScaledContents(True)

        # Menambahkan spasi antar kolom dan baris
        self.layout.setHorizontalSpacing(20)  # Jarak horizontal antara kamera
        self.layout.setVerticalSpacing(20)  # Jarak vertikal antara kamera

        self.setLayout(self.layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.previous_positions = {}

        # Folder untuk menyimpan hasil capture
        self.save_dir = "captured"
        os.makedirs(self.save_dir, exist_ok=True)

        self.prev_gray = None  # Untuk frame sebelumnya

        self.cooldown_active = False
        self.cooldown_start_time = 0
        self.last_draw_time = 0  # Timestamp terakhir bounding box dicetak

        # Load YOLOv8 model
        weight_path = os.path.abspath("Weights/yolov8.pt")
        self.model = YOLO(weight_path)
        self.class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']

        weight_path2 = os.path.abspath("Weights/yolov8Container.pt")
        self.model2 = YOLO(weight_path2)
        self.class_labels2 = ['0', '1', '2']

    def generate_black_qimage(self, width, height):
        black_frame = np.zeros(
            (height, width, 3), dtype=np.uint8)  # Hitam penuh
        rgb_black = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb_black.data, width, height, width * 3, QImage.Format_RGB888)

    def ocr_image_to_text(self, image_path):
        reader = easyocr.Reader(['en'], gpu=False)
        img = cv2.imread(image_path)
        result = reader.readtext(img)
        for detection in result:
            print(f"[text {detection}]:")
        # Load image
        # img = cv2.imread(image_path)
        # if img is None:
        #     print(f"Gagal membaca gambar dari path: {image_path}")
        #     return ""

        # # Get image dimensions
        # height, width = img.shape[:2]

        # # Crop the right half of the image
        # right_half = img[:, width//2:, :]

        # # Zoom the right half by scaling it up by 2x
        # zoom_factor = 2
        # zoomed_right_half = cv2.resize(
        #     right_half, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

        # # Create a visualization image (original with right side highlighted)
        # vis_img = img.copy()
        # # Draw a green rectangle on the right half to show the OCR detection area
        # cv2.rectangle(vis_img, (width//2, 0), (width, height), (0, 255, 0), 2)

        # valid_words = ['22G1', '22G0', '45R1', '22T1']

        # # instance text detector
        # reader = easyocr.Reader(['en'], gpu=False)
        # # detect text on zoomed right half of image
        # text_ = reader.readtext(zoomed_right_half)
        # threshold = 0.25

        # # Collect all detected text
        # all_detected_text = []

        # # Sort text items by their y-coordinate (top to bottom)
        # text_.sort(key=lambda x: x[0][0][1])  # Sort by top-left y-coordinate

        # # draw text on the visualization image
        # for i, t in enumerate(text_):
        #     # print(t)
        #     bbox, text, score = t
        #     # 1. OCR asli
        #     text_original = text.upper()
        #     # 2. OCR setelah ganti 6 -> G
        #     text_replaced = text_original.replace('6', 'G')

        #     # 3. Cari kemiripan dengan valid_words
        #     match_orig = difflib.get_close_matches(
        #         text_original, valid_words, n=1, cutoff=0.8)
        #     match_fixed = difflib.get_close_matches(
        #         text_replaced, valid_words, n=1, cutoff=0.8)

        #     # 4. Pilih hasil yang paling cocok
        #     if match_fixed and (not match_orig or match_fixed[0] != match_orig[0]):
        #         final_text = match_fixed[0]
        #     elif match_orig:
        #         final_text = match_orig[0]
        #     else:
        #         final_text = text_original  # Tidak cocok, pakai hasil OCR asli

        #     # Clean the text - remove square brackets
        #     cleaned_text = re.sub(r'[\[\]]', '', final_text)

        #     # Store cleaned text (we'll process it after the loop)
        #     if cleaned_text:
        #         all_detected_text.append((i, cleaned_text, bbox))

        #     if score > threshold:
        #         # Adjust bounding box coordinates to match the original image
        #         # Need to scale back down and offset for the right half
        #         bbox_adjusted = [
        #             (bbox[0][0]/zoom_factor + width//2,
        #              bbox[0][1]/zoom_factor),  # top-left
        #             (bbox[1][0]/zoom_factor + width//2,
        #              bbox[1][1]/zoom_factor),  # top-right
        #             (bbox[2][0]/zoom_factor + width//2,
        #              bbox[2][1]/zoom_factor),  # bottom-right
        #             (bbox[3][0]/zoom_factor + width//2,
        #              bbox[3][1]/zoom_factor)   # bottom-left
        #         ]

        #         # Draw rectangle and text on original image
        #         cv2.rectangle(vis_img,
        #                       (int(bbox_adjusted[0][0]),
        #                        int(bbox_adjusted[0][1])),
        #                       (int(bbox_adjusted[2][0]),
        #                        int(bbox_adjusted[2][1])),
        #                       (0, 0, 255), 2)
        #         cv2.putText(vis_img, cleaned_text,
        #                     (int(bbox_adjusted[0][0]), int(
        #                         bbox_adjusted[0][1]) - 10),
        #                     cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 0, 0), 2)

        # # Only combine the first 3 lines, keep the rest separate
        # first_three = [text for i, text, _ in all_detected_text[:3] if text]
        # combined_text = ''.join(first_three)
        print("Container Number:")

    def is_container_present(self, frame):
        # Simpan frame sementara untuk deteksi
        temp_path = os.path.join(self.save_dir, "temp_cam1.jpg")
        cv2.imwrite(temp_path, frame)

        # Lakukan deteksi dengan YOLOv8
        results = self.model2(temp_path)[0]

        # Cek apakah ada label '0 atau 1'
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.class_labels2[cls_id] if cls_id < len(
                self.class_labels2) else "Unknown"
            if label.lower() == "0" or label.lower() == "1":
                print("Validasi sukses: Container terdeteksi")
                return True
        print("Validasi gagal: Bukan container")
        return False

    def detect_damage_yolo(self, image_path,  camera_id=None):
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
        annotated_path = image_path.replace(
            ".jpg", f"_cam{camera_id}_detected.jpg")
        cv2.imwrite(annotated_path, img)
        # Logging
        print(f"[Kamera {camera_id}] Hasil deteksi disimpan: {annotated_path}")
        print(
            f"[Kamera {camera_id}] Total Damage: {sum(damage_counter.values())}")
        print(f"[Kamera {camera_id}] Per kategori:")
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
            cv2.putText(frame1, "Kamera 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 255, 0), 1, cv2.LINE_AA)

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

                    # Validasi dulu apakah benar container
                    if not self.is_container_present(clean_frame1):
                        print("Deteksi dibatalkan: Bukan container")
                        return

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

                    # if ret2:
                    #     cv2.imwrite(filename2, frame2)
                    # if ret3:
                    #     cv2.imwrite(filename3, frame3)
                    # if ret4:
                    #     cv2.imwrite(filename4, frame4)

                    self.detect_damage_yolo(filename1, camera_id=1)
                    # self.detect_damage_yolo(filename2, camera_id=2)
                    # self.detect_damage_yolo(filename3, camera_id=3)
                    # self.detect_damage_yolo(filename4, camera_id=4)
                    self.ocr_image_to_text(filename1)
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
            cv2.putText(frame2, "Kamera 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3, cv2.LINE_AA)

            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            qimg2 = QImage(rgb2.data, width2, height2,
                           width2 * 3, QImage.Format_RGB888)
            self.label2.setPixmap(QPixmap.fromImage(qimg2))
        else:
            black_img = self.generate_black_qimage(640, 480)
            self.label2.setPixmap(QPixmap.fromImage(black_img))

        if ret3:
            height3, width3, _ = frame3.shape
            cv2.putText(frame3, "Kamera 3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3, cv2.LINE_AA)
            rgb3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            qimg3 = QImage(rgb3.data, width3, height3,
                           width3 * 3, QImage.Format_RGB888)
            self.label3.setPixmap(QPixmap.fromImage(qimg3))
        else:
            black_img = self.generate_black_qimage(640, 480)
            self.label3.setPixmap(QPixmap.fromImage(black_img))

        if ret4:
            height4, width4, _ = frame4.shape
            cv2.putText(frame4, "Kamera 4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 1, cv2.LINE_AA)
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
