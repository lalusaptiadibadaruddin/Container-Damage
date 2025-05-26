import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget, QVBoxLayout, QLabel
import time
import datetime
import jwt
import os
import json
import requests
import random
import string
from ultralytics import YOLO
from collections import Counter
import difflib
import easyocr
import re


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # RTSP URL kamera
        # url = "rtsp://itdev:itdev@2025@192.168.160.54:554/cam/realmonitor?channel=1&subtype=0"
        # url2 = "rtsp://itdev:itdev@2025@192.168.160.54:554/cam/realmonitor?channel=2&subtype=0"
        # url3 = "rtsp://itdev:itdev@2025@192.168.160.54:554/cam/realmonitor?channel=3&subtype=0"

        self.cap1 = cv2.VideoCapture(0)
        # self.cap2 = cv2.VideoCapture(url)
        # self.cap3 = cv2.VideoCapture(url2)
        # self.cap4 = cv2.VideoCapture(url3)

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
        # self.label2_name = QLabel("Kamera 2", self)
        # self.label3_name = QLabel("Kamera 3", self)
        # self.label4_name = QLabel("Kamera 4", self)

        self.label1 = QLabel(self)
        # self.label2 = QLabel(self)
        # self.label3 = QLabel(self)
        # self.label4 = QLabel(self)

        # Menambahkan nama kamera dan frame ke dalam layout
        self.layout.addWidget(self.label1, 0, 0)
        # self.layout.addWidget(self.label2, 0, 1)
        # self.layout.addWidget(self.label3, 1, 0)
        # self.layout.addWidget(self.label4, 1, 1)

        self.label1.setScaledContents(True)
        # self.label2.setScaledContents(True)
        # self.label3.setScaledContents(True)
        # self.label4.setScaledContents(True)

        # Menambahkan spasi antar kolom dan baris
        self.layout.setHorizontalSpacing(20)  # Jarak horizontal antara kamera
        self.layout.setVerticalSpacing(20)  # Jarak vertikal antara kamera

        self.setLayout(self.layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Load YOLOv8 model
        weight_path = os.path.abspath("Weights/yolov8.pt")
        self.model = YOLO(weight_path)
        self.class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']
        weight_path2 = os.path.abspath("Weights/yolov8Container.pt")
        self.model2 = YOLO(weight_path2)
        self.class_labels2 = ['0', '1', '2']

        # Folder untuk menyimpan hasil capture
        self.save_dir = "captured"
        os.makedirs(self.save_dir, exist_ok=True)

        self.upload_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),  # folder tempat script ini
                'monitoring-container-damage-be',
                'uploads',
                'manual-scan'
            )
        )

        # os.makedirs(self.upload_dir, exist_ok=True)

        self.prev_gray = None  # Untuk frame sebelumnya

        self.cooldown_active = False
        self.cooldown_start_time = 0
        self.capture_after_cooldown = False
        self.last_detection_time = 0
        self.cooldown_duration = 10  # dalam detik
        self.object_touching_line = False  # status objek saat ini
        self.processing = False  # Lock untuk proses satu container
        # self.last_draw_time = 0  # Timestamp terakhir bounding box dicetak

    # generate black bacground

    def generate_black_qimage(self, width, height):
        black_frame = np.zeros(
            (height, width, 3), dtype=np.uint8)  # Hitam penuh
        rgb_black = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb_black.data, width, height, width * 3, QImage.Format_RGB888)

    # Random untuk nomor container (misal: 4 huruf + 4 angka)
    def generate_random_container_number(self):
        return "CONT" + ''.join(random.choices(string.digits, k=6))

    # Random untuk tipe container (misal: 4 huruf)
    def generate_random_container_type(self):
        digits = ''.join(random.choices(string.digits, k=2))
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        return digits + letters

    def ocr_image_to_text(self, image_path):
        combined_text_no_container = combined_text_type_container = ""
        # # reader = easyocr.Reader(['en'], gpu=False)
        # # Load image
        # img = cv2.imread(image_path)

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

        #     # print(match_fixed)

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

        #     # print(all_detected_text[:5])

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
        # first_three = [text for i, text, _ in all_detected_text[:5] if text]
        # # print(first_three)

        # combined_text = ''.join(first_three)
        # combined_text_no_container = combined_text[0:11]
        # combined_text_type_container = combined_text[11:15]

        combined_text_no_container = self.generate_random_container_number()
        combined_text_type_container = "22G2"

        return combined_text_no_container, combined_text_type_container

    # validasi container
    def is_container_present(self, frame):
        # Simpan frame sementara untuk deteksi
        temp_path = os.path.join(self.save_dir, "temp_cam1.jpg")
        cv2.imwrite(temp_path, frame)

        # Deteksi dengan YOLOv8
        results = self.model2(temp_path)[0]

        container_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.class_labels2[cls_id] if cls_id < len(
                self.class_labels2) else "Unknown"
            conf = box.conf[0] if hasattr(box, 'conf') else 0

            # Ambil koordinat bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Gambar bounding box dan label
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # text = f"{label} {conf:.2f}"
            # cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 255, 255), 2, cv2.LINE_AA)

            # Cek label container
            if label.lower() in ["0", "1"]:
                container_detected = True

        # Jika container terdeteksi, simpan hasil deteksi dengan bounding box
        if container_detected:
            print("Validasi sukses: Container terdeteksi")
            result_path = os.path.join(self.save_dir, "detected_cam1.jpg")
            cv2.imwrite(result_path, frame)
            print(f"Hasil deteksi disimpan ke: {result_path}")

        return container_detected

    def generate_jwt_token(self):
        # JWT secret from .env (same as in your Node.js app)
        jwt_secret = "cef16827b7a3116309dfadecc811629eb2303056227cfebac15a1c48454ca961"

        # Create payload with expiration time (e.g., 1 hour)
        payload = {
            # You can use any identifier that makes sense
            "userId": "container-damage-detector",
            "iat": datetime.datetime.utcnow(),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }

        # Generate the token
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")
        return token

    # deteksi damage container
    def detect_damage_yolo(self, image_path, prefix):
        results = self.model(image_path)[0]
        # Inisialisasi dengan semua label dan nilai 0
        damage_counter = {label: 0 for label in self.class_labels}
        # Baca ulang gambar
        img = cv2.imread(image_path)
        # damage_counter = Counter()
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = self.class_labels[cls_id] if cls_id < len(
                self.class_labels) else "Unknown"
            conf = float(box.conf[0])

            # Tambah jumlah kategori
            damage_counter[label] += 1

            # Gambar bounding box dan label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Tambahkan info jumlah di gambar
        y_offset = 20
        total_damage = sum(damage_counter.values())
        cv2.putText(img, f"Total Damage: {total_damage}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        for label in self.class_labels:
            count = damage_counter[label]
            cv2.putText(img, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # posisi vertikal untuk menampilkan teks (agar tidak tumpang tindih antar label).
            y_offset += 25

        # Simpan hasil dengan anotasi
        annotated_path = image_path.replace(
            ".jpg", f"{prefix}_detected.jpg")
        cv2.imwrite(annotated_path, img)

        # Mapping label
        label_mapping = {
            "Karat": "rust",
            "Lubang": "hole",
            "Patah": "broken",
            "Penyok": "dents",
            "Retak": "crack"
        }
        # Buat list kategori dalam format yang diinginkan
        categories = []
        for label, count in damage_counter.items():
            if count > 0:
                categories.append({
                    "name": label_mapping[label],
                    "damageSeverity": "1",  # bisa diubah sesuai logika
                    "damageTotal": count
                })

        return {
            "image_path": annotated_path,
            "detail": {
                "damageLocation": prefix,
                "categories": categories
            }
        }

        # Logging
        # print(f"[Kamera {camera_id}] Hasil deteksi disimpan: {annotated_path}")
        # print(
        #     f"[Kamera {camera_id}] Total Damage: {sum(damage_counter.values())}")
        # print(f"[Kamera {camera_id}] Per kategori:")
        # print("Per kategori:")
        # for label in self.class_labels:
        #     print(f" - {label}: {damage_counter[label]}")

        # self.send_to_api(camera_id, damage_counter, annotated_path)

    def get_best_focus_frame(self, num_frames=5):
        best_score = 0
        best_frame = None
        for _ in range(num_frames):
            ret, frame = self.cap1.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
            time.sleep(0.05)  # Beri jeda kecil antar frame
        return best_frame

    # loop deteksi real-time

    def update_frame(self):
        ret1, frame1 = self.cap1.read()
        # ret2, frame2 = self.cap2.read()
        # ret3, frame3 = self.cap3.read()
        # ret4, frame4 = self.cap4.read()

        # background = self.background
        if ret1:
            image_data = {"Back": None, "Left": None,
                          "Top": None, "Right": None}
            clean_frame1 = frame1.copy()
            height1, width1, _ = frame1.shape
            cv2.putText(frame1, "Kamera 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 255, 0), 1, cv2.LINE_AA)

            # center_x = width1 // 2
            center_x = width1 - 50
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

            current_time = time.time()

            # Jika sedang cooldown, tunggu sampai selesai
            if self.cooldown_active:
                if current_time - self.cooldown_start_time >= 10:
                    print("Cooldown Selesai!")
                    self.cooldown_active = False
                    self.capture_after_cooldown = True  # Set flag untuk capture setelah cooldown
                else:
                    rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    qimg1 = QImage(rgb1.data, width1, height1,
                                   width1 * 3, QImage.Format_RGB888)
                    self.label1.setPixmap(QPixmap.fromImage(qimg1))
                    return

            # Perulangan semua kontur
            for contour in cnts:
                if cv2.contourArea(contour) < 20000:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                cx = x + w // 2

                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if x <= center_x <= x + w:
                    print("Bounding Box Touched the Line!")
                    self.cooldown_active = True
                    self.cooldown_start_time = current_time
                    break  # hanya perlu satu yang menyentuh garis

            # Tangkap gambar terbaik setelah cooldown selesai
            if hasattr(self, 'capture_after_cooldown') and self.capture_after_cooldown and not self.processing:
                self.processing = True
                print("Capture setelah cooldown selesai")
                time.sleep(0.3)  # Delay pendek sebelum capture
                frame_to_save = self.get_best_focus_frame(num_frames=5)

                # Validasi container
                if not self.is_container_present(frame_to_save):
                    print("validasi Gagal, Bukan container")
                else:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename1 = os.path.join(
                        self.save_dir, f"back_{timestamp}.jpg")
                    cv2.imwrite(filename1, frame_to_save)

                    basename = os.path.basename(filename1)
                    prefix = basename.split('_')[0].capitalize()
                    print(prefix)

                    if prefix == "Back":
                        container_number, container_type = self.ocr_image_to_text(
                            filename1)

                    image_data[prefix] = self.detect_damage_yolo(
                        filename1, prefix)

                if not image_data.get("Back"):
                    print("Gambar Back harus tersedia.")
                    self.capture_after_cooldown = False
                    self.processing = False  # Lepas lock
                    return

                details_list = []

                if image_data["Back"]:
                    details_list.append(image_data["Back"]["detail"])
                if image_data["Left"]:
                    details_list.append(image_data["Left"]["detail"])
                if image_data["Top"]:
                    details_list.append(image_data["Top"]["detail"])
                if image_data["Right"]:
                    details_list.append(image_data["Right"]["detail"])

                files = []
                for side in ["Back", "Left", "Top", "Right"]:
                    if image_data[side] and "image_path" in image_data[side]:
                        image_path = image_data[side]["image_path"]
                        files.append((
                            'images',
                            (os.path.basename(image_path), open(
                                image_path, 'rb'), 'image/jpeg')
                        ))

                # Generate JWT token
                token = self.generate_jwt_token()
                # print(token)
                url = "http://localhost:3000/api/containers/auto-scan"  # Ganti dengan URL API-mu
                headers = {
                    'Authorization': f'Bearer {token}'
                }

                payload = {
                    "no_container": container_number,
                    "container_type": container_type,
                    "details": json.dumps(details_list)
                }

                print('payload:')
                print(payload)
                print('files:')
                print(files)

                # Kirim POST request
                response = requests.post(
                    url,
                    headers=headers,
                    data=payload,
                    files=files
                )

                for _, file_tuple in files:
                    file_tuple[1].close()

                if response.status_code == 201:
                    print("Data successfully sent to API")
                    print(response.json())

                    # Hapus semua file image hasil deteksi
                    for side in ["Back", "Left", "Top", "Right"]:
                        if image_data[side] and "image_path" in image_data[side]:
                            # Tunggu sejenak untuk pastikan file tidak lagi dipakai
                            time.sleep(0.5)
                            try:
                                os.remove(image_data[side]["image_path"])
                                print(
                                    f"Deleted: {image_data[side]['image_path']}")
                            except Exception as e:
                                print(
                                    f"Failed to delete {image_data[side]['image_path']}: {e}")
                else:
                    print(
                        f"Failed to send data. Status code: {response.status_code}")
                    print(response.text)

                self.capture_after_cooldown = False
                self.processing = False  # Lepas lock

            rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            qimg1 = QImage(rgb1.data, width1, height1,
                           width1 * 3, QImage.Format_RGB888)
            self.label1.setPixmap(QPixmap.fromImage(qimg1))
        else:
            black_img1 = self.generate_black_qimage(640, 480)
            self.label1.setPixmap(QPixmap.fromImage(black_img1))

        # if ret2:

        #     height2, width2, _ = frame2.shape
        #     cv2.putText(frame2, "Kamera 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (0, 255, 0), 3, cv2.LINE_AA)

        #     rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        #     qimg2 = QImage(rgb2.data, width2, height2,
        #                    width2 * 3, QImage.Format_RGB888)
        #     self.label2.setPixmap(QPixmap.fromImage(qimg2))
        # else:
        #     black_img = self.generate_black_qimage(640, 480)
        #     self.label2.setPixmap(QPixmap.fromImage(black_img))

        # if ret3:
        #     height3, width3, _ = frame3.shape
        #     cv2.putText(frame3, "Kamera 3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #                 2, (0, 255, 0), 3, cv2.LINE_AA)
        #     rgb3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        #     qimg3 = QImage(rgb3.data, width3, height3,
        #                    width3 * 3, QImage.Format_RGB888)
        #     self.label3.setPixmap(QPixmap.fromImage(qimg3))
        # else:
        #     black_img = self.generate_black_qimage(640, 480)
        #     self.label3.setPixmap(QPixmap.fromImage(black_img))

        # if ret4:
        #     height4, width4, _ = frame4.shape
        #     cv2.putText(frame4, "Kamera 4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #                 2, (0, 255, 0), 1, cv2.LINE_AA)
        #     rgb4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
        #     qimg4 = QImage(rgb4.data, width4, height4,
        #                    width4 * 3, QImage.Format_RGB888)
        #     self.label4.setPixmap(QPixmap.fromImage(qimg4))
        # else:
        #     black_img = self.generate_black_qimage(640, 480)
        #     self.label4.setPixmap(QPixmap.fromImage(black_img))

    def closeEvent(self, event):
        self.cap1.release()
        # self.cap2.release()
        # self.cap3.release()
        # self.cap4.release()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
