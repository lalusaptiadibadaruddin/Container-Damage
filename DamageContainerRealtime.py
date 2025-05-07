import cv2
import math
import cvzone
from ultralytics import YOLO
import os
from datetime import datetime

# Load YOLO model
yolo_model = YOLO("Weights/yolov8.pt")

# Daftar label class
class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']

# Buka kamera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Buat folder penyimpanan jika belum ada
save_folder = "Detected_Damage"
os.makedirs(save_folder, exist_ok=True)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Deteksi menggunakan YOLO
    # stream=True untuk real-time performance
    results = yolo_model(frame, stream=True)
    count = 0
    damage_counts = {label: 0 for label in class_labels}

    save_this_frame = False  # Flag untuk menyimpan frame

    for detection in results:
        boxes = detection.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf < 0.40:
                continue  # Lewati deteksi dengan confidence rendah

            cls = int(box.cls[0])
            if 0 <= cls < len(class_labels):  # untuk menghindari IndexError
                class_name = class_labels[cls]
                damage_counts[class_name] += 1

                # Simpan jika ditemukan "Karat", "Lubang", "Patah", "Penyok", "Retak"
                if class_name in ["Karat", "Lubang", "Patah", "Penyok", "Retak"] and conf > 0.40:
                    save_this_frame = True
                # Gambar kotak dan teks
                    cvzone.cornerRect(frame, (x1, y1, w, h), t=1)
                    cvzone.putTextRect(frame, f'{class_name} {conf}', (x1, y1 - 10),
                                       scale=0.8, thickness=1, colorR=(255, 0, 0))
                    count += 1

    # Simpan frame jika perlu
    if damage_counts["Penyok"] > 0 or damage_counts["Karat"] > 0 or damage_counts["Patah"] > 0 or damage_counts["Retak"] > 0 or damage_counts["Lubang"] > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"damage_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"[INFO] Gambar disimpan: {save_path}")

    # Tampilkan total count dan per class di frame
    cvzone.putTextRect(
        frame, f"Total Damage: {count}", (10, 30), scale=1, thickness=2, colorR=(0, 255, 0))
    y_offset = 60

    # Print total dan per kategori ke terminal

    print(f"\n[INFO] Jumlah Damage Terdeteksi: {count}")
    for label, jumlah in damage_counts.items():
        if jumlah > 0:
            cvzone.putTextRect(
                frame, f"{label}: {jumlah}", (10, y_offset), scale=0.8, thickness=1)
            y_offset += 30
            print(f" - {label}: {jumlah}")

    # Tampilkan frame
    cv2.imshow("Real-Time Container Damage Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
