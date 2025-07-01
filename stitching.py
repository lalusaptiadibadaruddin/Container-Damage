import cv2
from ultralytics import YOLO
import time
import os
import numpy as np

model = YOLO('yolov8n.pt')  # Ganti jika pakai model custom
cap = cv2.VideoCapture(0)

TRUCK_CLASS_ID = 7
capture_folder = "stitched_input"
stitched_folder = "stitched_output"
os.makedirs(capture_folder, exist_ok=True)
os.makedirs(stitched_folder, exist_ok=True)

captured_flags = {'zone1': False, 'zone2': False, 'zone3': False}
captured_files = {'zone1': '', 'zone2': '', 'zone3': ''}
stitched_done = False
cooldown_time = 5
last_stitch_time = 0


def stitch_images(image_paths, output_path):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    if len(images) == 3:
        min_height = min(img.shape[0] for img in images)
        resized_images = [cv2.resize(
            img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]
        stitched = np.hstack(resized_images)
        cv2.imwrite(output_path, stitched)
        print(f"[STITCHED] Disimpan di {output_path}")
        return True
    return False


def delete_files(file_list):
    for path in file_list:
        if os.path.exists(path):
            os.remove(path)
            print(f"[DELETE] {path} dihapus.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    zone1 = width // 4
    zone2 = width // 2
    zone3 = (3 * width) // 4

    original_frame = frame.copy()
    results = model(frame)[0]
    touched_zones = set()

    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy())
        conf = box.conf.cpu().numpy()[0]

        if cls_id == TRUCK_CLASS_ID and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

            if zone1 <= x2 < zone2:
                touched_zones.add('zone1')
            elif zone2 <= x2 < zone3:
                touched_zones.add('zone2')
            elif x2 >= zone3:
                touched_zones.add('zone3')

            label = f'Truck {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for zone in ['zone1', 'zone2', 'zone3']:
        if zone in touched_zones and not captured_flags[zone]:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(
                capture_folder, f"capture_{zone}_{timestamp}.jpg")

            # Crop zona1 dari original frame meskipun menyentuh zona2/zona3
            zona1_crop = original_frame[:, 0:zone1]
            cv2.imwrite(filename, zona1_crop)
            print(f"[CAPTURE] {zone} -> zona1 disimpan: {filename}")

            captured_flags[zone] = True
            captured_files[zone] = filename

    # Gambar garis hanya jika zona belum disentuh
    if not captured_flags['zone1']:
        cv2.line(frame, (zone1, 0), (zone1, height), (0, 0, 255), 2)
    if not captured_flags['zone2']:
        cv2.line(frame, (zone2, 0), (zone2, height), (0, 0, 255), 2)
    if not captured_flags['zone3']:
        cv2.line(frame, (zone3, 0), (zone3, height), (0, 0, 255), 2)

    # Stitching jika sudah dapat semua zona
    if all(captured_flags.values()) and not stitched_done:
        stitched_filename = os.path.join(
            stitched_folder, f"stitched_output_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        )
        success = stitch_images(
            [captured_files['zone3'], captured_files['zone2'], captured_files['zone1']],
            stitched_filename
        )
        if success:
            # delete_files([captured_files['zone1'],
            #              captured_files['zone2'], captured_files['zone3']])
            stitched_done = True
            last_stitch_time = time.time()

    # Reset siklus setelah cooldown
    if stitched_done and (time.time() - last_stitch_time) > cooldown_time:
        captured_flags = {zone: False for zone in captured_flags}
        captured_files = {zone: '' for zone in captured_files}
        stitched_done = False
        print("[RESET] Siap untuk capture selanjutnya.")

    cv2.imshow('Truck Detection YOLOv8', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
