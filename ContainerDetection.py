import cv2
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model folder weights
yolo_model = YOLO("Weights/yolov8.pt")

# yolo_model = YOLO("Weights/yolov8image.pt")

# definisi class
class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']
# print(class_labels)

# load image
image_path = "Media/Input2/container1 (4).jpg"
img = cv2.imread(image_path)

# detection yolo
results = yolo_model(img)
count = 0
# dictionary untuk menghitung jumlah per class = 0
damage_counts = {label: 0 for label in class_labels}

# perulangan result
for detection in results:
    # mengambil bounding boxes
    boxes = detection.boxes
    # print(boxes)
    # perulangan boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        # print(box.xyxy[0])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        # confidence score
        # print(box.conf[0])

        conf = math.ceil((box.conf[0] * 100)) / 100

        cls = int(box.cls[0])

        # Tambahan: Hitung jumlah per class
        class_name = class_labels[cls]
        damage_counts[class_name] += 1

        # if conf > 0.3:
        # membuat kotak font dan color
        cvzone.cornerRect(img, (x1, y1, w, h), t=1)
        cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (
            x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))
        count = count + 1


jumlah_damage = f"Jumlah Damage {count}"
print(jumlah_damage)
# Print jumlah per class
for label, jumlah in damage_counts.items():
    print(f"{label}: {jumlah}")

# Resize gambar ke ukuran yang diinginkan, misal 800x600
img_resized = cv2.resize(img, (900, 700))  # (width, height)
# tampilkan image detection
cv2.imshow("Image", img_resized)
# keluar
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.waitKey(1)
