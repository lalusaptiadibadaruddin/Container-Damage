import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import easyocr
import re
import difflib
import json

model = YOLO('yolov8n.pt')
damage_model = YOLO("Weights/yolov8.pt")  # Ganti dengan model kerusakanmu

reader = easyocr.Reader(['en'])
# Kamera: 1 = Trigger, 2-4 = Capture, 5 = Back
cap_trigger = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)  # Left
cap3 = cv2.VideoCapture(2)  # Top
cap4 = cv2.VideoCapture(3)  # Right
cap5 = cv2.VideoCapture(4)  # Back

TRUCK_CLASS_ID = 7
capture_folder = "stitched_input"
stitched_folder = "stitched_output"
os.makedirs(capture_folder, exist_ok=True)
os.makedirs(stitched_folder, exist_ok=True)

captured_flags = {'zone1': False, 'zone2': False, 'zone3': False}
captured_files = {
    'zone1': {'left': '', 'top': '', 'right': ''},
    'zone2': {'left': '', 'top': '', 'right': ''},
    'zone3': {'left': '', 'top': '', 'right': ''}
}

# definisi class
class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']

back_captured = False
back_file = ''

stitched_done = False
cooldown_time = 5
last_stitch_time = 0
capture_start_time = None
capture_timeout = 30


def stitch_images_per_camera(captured_files, cam_name, output_dir):
    paths = [
        captured_files['zone3'][cam_name],
        captured_files['zone2'][cam_name],
        captured_files['zone1'][cam_name]
    ]
    images = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARNING] File tidak ditemukan: {path}")
            return False
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Gagal membaca gambar: {path}")
            return False
        images.append(img)

    # print(len(images))
    if len(images) == 3:
        min_height = min(img.shape[0] for img in images)
        resized_images = [cv2.resize(
            img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]
        stitched = np.hstack(resized_images)
        output_path = os.path.join(
            output_dir, f"{cam_name}-{time.strftime('%Y%m%d-%H%M%S')}.jpg")
        cv2.imwrite(output_path, stitched)
        print(f"[STITCHED] {cam_name.upper()} disimpan di {output_path}")
        return True
    else:
        print("[ERROR] Jumlah gambar kurang dari 3 atau lebih dari 3")
        return False


def delete_files(file_list):
    for path in file_list:
        if os.path.exists(path):
            os.remove(path)
            print(f"[DELETE] {path} dihapus.")


def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return frame


def detect_damage(image_path, model, prefix=''):
    img = cv2.imread(image_path)
    if img is None:
        print(
            f"[ERROR] Tidak bisa membaca gambar {prefix} untuk deteksi kerusakan.")
        return

    results = model(image_path)[0]
    # Inisialisasi dengan semua label dan nilai 0
    damage_counter = {label: 0 for label in class_labels}
    # Baca ulang gambar
    img = cv2.imread(image_path)

    # damage_counter = Counter()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = class_labels[cls_id] if cls_id < len(
            class_labels) else "Unknown"
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

    for label in class_labels:
        count = damage_counter[label]
        cv2.putText(img, f"{label}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # posisi vertikal untuk menampilkan teks (agar tidak tumpang tindih antar label).
        y_offset += 25

    # Simpan hasil dengan anotasi
    annotated_path = image_path.replace(
        ".jpg", f"_detected.jpg")

    cv2.imwrite(annotated_path, img)

    # Mapping label
    label_mapping = {
        "Karat": "rust",
        "Lubang": "holes",
        "Patah": "breaks",
        "Penyok": "dents",
        "Retak": "cracks"
    }
    # Buat list kategori dalam format yang diinginkan
    categories = []
    for label in class_labels:
        mapped_name = label_mapping.get(label, label.lower())
        count = damage_counter[label]

        categories.append({
            "name": mapped_name,
            "damageSeverity": "1" if count > 0 else "0",
            "damageTotal": count
        })

    return {
        "original_path": image_path,
        "image_path": annotated_path,
        "detail": {
            "damageLocation": prefix,
            "categories": categories
        }
    }


def ocr_image_to_text(image_path, reader, cam_name=''):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Tidak bisa membaca gambar {cam_name} untuk OCR.")
        return

    # Get image dimensions
    height, width = img.shape[:2]

    # Crop the right half of the image
    right_half = img[:, width//2:, :]

    # Zoom the right half by scaling it up by 2x
    zoom_factor = 2
    zoomed_right_half = cv2.resize(
        right_half, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    # Create a visualization image (original with right side highlighted)
    vis_img = img.copy()
    # Draw a green rectangle on the right half to show the OCR detection area
    cv2.rectangle(vis_img, (width//2, 0), (width, height), (0, 255, 0), 2)

    valid_words = ['22G1', '22G0', '45R1', '22T1', '42G1', '45G1']

    # detect text on zoomed right half of image
    text_ = reader.readtext(zoomed_right_half)
    threshold = 0.25
    # Collect all detected text
    all_detected_text = []

    # Sort text items by their y-coordinate (top to bottom)
    text_.sort(key=lambda x: x[0][0][1])  # Sort by top-left y-coordinate

    # draw text on the visualization image
    for i, t in enumerate(text_):
        # print(t)
        bbox, text, score = t
        # 1. OCR asli
        text_original = text.upper()
        # 2. OCR setelah ganti 6 -> G
        text_replaced = text_original.replace('6', 'G')

        # 3. Cari kemiripan dengan valid_words
        match_orig = difflib.get_close_matches(
            text_original, valid_words, n=1, cutoff=0.8)

        match_fixed = difflib.get_close_matches(
            text_replaced, valid_words, n=1, cutoff=0.8)

        # 4. Pilih hasil yang paling cocok
        if match_fixed and (not match_orig or match_fixed[0] != match_orig[0]):
            final_text = match_fixed[0]
        elif match_orig:
            final_text = match_orig[0]
        else:
            final_text = text_original  # Tidak cocok, pakai hasil OCR asli

        # Clean the text - remove square brackets
        cleaned_text = re.sub(r'[\[\]]', '', final_text)

        # Store cleaned text (we'll process it after the loop)
        if cleaned_text:
            all_detected_text.append((i, cleaned_text, bbox))

        # print(all_detected_text)
        if score > threshold:
            # Adjust bounding box coordinates to match the original image
            # Need to scale back and offset for the right half
            bbox_adjusted = [
                (bbox[0][0]/zoom_factor + width//2,
                 bbox[0][1]/zoom_factor),  # top-left
                (bbox[1][0]/zoom_factor + width//2,
                 bbox[1][1]/zoom_factor),  # top-right
                (bbox[2][0]/zoom_factor + width//2,
                 bbox[2][1]/zoom_factor),  # bottom-right
                (bbox[3][0]/zoom_factor + width//2,
                 bbox[3][1]/zoom_factor)   # bottom-left
            ]

        # Draw rectangle and text on original image
            cv2.rectangle(vis_img,
                          (int(bbox_adjusted[0][0]),
                           int(bbox_adjusted[0][1])),
                          (int(bbox_adjusted[2][0]),
                           int(bbox_adjusted[2][1])),
                          (0, 0, 255), 2)
            cv2.putText(vis_img, cleaned_text,
                        (int(bbox_adjusted[0][0]), int(
                            bbox_adjusted[0][1]) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 0, 0), 2)

     # Only combine the first 3 lines, keep the rest separate
    first_three = [text for i, text,
                   _ in all_detected_text[:5] if text]
    # print(first_three)
    text = ''.join(first_three)
    combined_text = ''.join(text.split())

    combined_text_no_container = combined_text[0:11]
    # print(combined_text_no_container)
    combined_text_type_container = combined_text[11:15]
    # print(combined_text_type_container)

    return combined_text_no_container, combined_text_type_container


while True:
    frame_trigger = get_frame(cap_trigger)
    height, width = frame_trigger.shape[:2]
    zone1 = width // 4
    zone2 = width // 2
    zone3 = (3 * width) // 4

    # original_frame = frame_trigger.copy()
    results = model(frame_trigger)[0]
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
            cv2.rectangle(frame_trigger, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_trigger, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            camera_mapping = {
                'left': cap2,
                'top': cap3,
                'right': cap4
            }

            for zone in ['zone1', 'zone2', 'zone3']:
                if zone in touched_zones and not captured_flags[zone]:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    for cam_name, capX in camera_mapping.items():
                        filename = os.path.join(
                            capture_folder, f"{cam_name}-{zone}-{timestamp}.jpg")
                        frame = get_frame(capX)
                        cv2.imwrite(filename, frame)
                        print(
                            f"[CAPTURE] {cam_name.upper()} - {zone} -> {filename}")
                    captured_flags[zone] = True
                    captured_files[zone] = {
                        cam: os.path.join(
                            capture_folder, f"{cam}-{zone}-{timestamp}.jpg")
                        for cam in camera_mapping
                    }

    # Capture kamera belakang saat zona3 tersentuh (hanya sekali)
    if 'zone3' in touched_zones and not back_captured:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        frame_back = get_frame(cap5)
        back_file = os.path.join(
            capture_folder, f"back-{timestamp}.jpg")
        cv2.imwrite(back_file, frame_back)
        print(f"[CAPTURE] BACK dari kamera5 -> disimpan: {back_file}")
        back_captured = True

    # any(captured_flags.values()) akan mengembalikan True
    if any(captured_flags.values()) and capture_start_time is None:
        capture_start_time = time.time()

    # capture_start_time is not None proses capture sudah dimulai (timer sedang berjalan).
    # (time.time() - capture_start_time > capture_timeout) waktu yang sudah berlalu sejak capture dimulai melebihi batas waktu yang diizinkan
    # not all(captured_flags.values()) belum semua zona berhasil dicapture
    ''' Jika proses capture sedang berjalan, dan sudah melewati batas waktu yang ditentukan, namun belum semua zona berhasil dicapture, maka lakukan reset atau abaikan proses'''
    if capture_start_time is not None and (time.time() - capture_start_time > capture_timeout) and not all(captured_flags.values()):
        print(
            "[TIMEOUT] Tidak semua zona tercapture. Menghapus file yang sudah tercapture.")
        # delete_files([path for path in captured_files.values() if path])
        captured_flags = {zone: False for zone in captured_flags}
        captured_files = {
            zone: {'left': '', 'top': '', 'right': ''}
            for zone in captured_files
        }
        capture_start_time = None
        back_captured = False
        back_file = ''

    # Garis zona
    # Cek apakah zona1 belum berhasil dicapture.
    # not captured_flags['zone1'] bernilai True
    if not captured_flags['zone1']:
        cv2.line(frame_trigger, (zone1, 0), (zone1, height), (0, 0, 255), 2)
        cv2.putText(frame_trigger, "Zone 1", (zone1 - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Cek apakah zona2 belum berhasil dicapture.
    if not captured_flags['zone2']:
        cv2.line(frame_trigger, (zone2, 0), (zone2, height), (0, 0, 255), 2)
        cv2.putText(frame_trigger, "Zone 2", (zone2 - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Cek apakah zona3 belum berhasil dicapture.
    if not captured_flags['zone3']:
        cv2.line(frame_trigger, (zone3, 0), (zone3, height), (0, 0, 255), 2)
        cv2.putText(frame_trigger, "Zone 3", (zone3 - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    ''' Jika semua zona sudah berhasil dicapture dan proses stitching belum dilakukan, maka lakukan proses stitching sekarang. '''
    # all(captured_flags.values()) cek semua nilai di dictionary captured_flags adalah True.
    if all(captured_flags.values()) and not stitched_done:
        success_left = stitch_images_per_camera(
            captured_files, 'left', stitched_folder)
        success_top = stitch_images_per_camera(
            captured_files, 'top', stitched_folder)
        success_right = stitch_images_per_camera(
            captured_files, 'right', stitched_folder)

        # Simpan back image juga
        if back_captured and os.path.exists(back_file):
            dst_back = os.path.join(stitched_folder, f"back-{timestamp}.jpg")
            cv2.imwrite(dst_back, cv2.imread(back_file))
            print(f"[STITCHED] BACK disimpan di {dst_back}")

        if success_left and success_top and success_right:
            # Hapus semua file capture setelah stitching sukses
            # delete_files([
            #     captured_files[z]['left']
            #     for z in captured_files
            # ] + [
            #     captured_files[z]['top']
            #     for z in captured_files
            # ] + [
            #     captured_files[z]['right']
            #     for z in captured_files
            # ])
            stitched_done = True
            last_stitch_time = time.time()
            # Deteksi kerusakan untuk stitched images
            # Inisialisasi dictionary hasil akhir
            detection_results = {
                "no_container": "",       # akan diisi nanti
                "container_type": "",     # akan diisi nanti
                "details": []             # list berisi hasil dari detect_damage
            }

            # Klasifikasi file berdasarkan prefix
            # files_by_prefix = {'left': [], 'top': [], 'right': [], 'back': []}
            # for file in os.listdir(stitched_folder):
            #     if file.endswith('.jpg'):
            #         for prefix in files_by_prefix.keys():
            #             if file.startswith(prefix):
            #                 files_by_prefix[prefix].append(file)

            files_by_prefix = {'left': [], 'top': [], 'right': [], 'back': []}
            for file in os.listdir(stitched_folder):
                if file.lower().endswith('.jpg'):
                    for prefix in files_by_prefix:
                        if file.startswith(f"{prefix}-"):
                            files_by_prefix[prefix].append(file)
                            break

            # Proses left, top, right
            for prefix in ['left', 'top', 'right']:
                for file in files_by_prefix[prefix]:
                    result = detect_damage(os.path.join(
                        stitched_folder, file), damage_model, prefix)
                    detection_results["details"].append(result["detail"])

            # Proses back (pastikan hanya satu atau ambil yang terbaru)
            if files_by_prefix["back"]:
                # ambil yang terakhir (atau pertama)
                back_file = sorted(files_by_prefix["back"])[-1]
                back_path = os.path.join(stitched_folder, back_file)

                result = detect_damage(back_path, damage_model, 'back')
                detection_results["details"].append(result["detail"])

                container_number, container_type = ocr_image_to_text(
                    back_path, reader, 'back')
                detection_results["no_container"] = container_number
                detection_results["container_type"] = container_type

            # Tampilkan hasil
            print(json.dumps(detection_results, indent=2))

        else:
            print("Stitching gagal. Menghapus file capture.")
            # delete_files([
            #     captured_files[z]['left']
            #     for z in captured_files
            # ] + [
            #     captured_files[z]['top']
            #     for z in captured_files
            # ] + [
            #     captured_files[z]['right']
            #     for z in captured_files
            # ])
            captured_flags = {zone: False for zone in captured_flags}
            captured_files = {
                zone: {'left': '', 'top': '', 'right': ''}
                for zone in captured_files
            }
            capture_start_time = None
            back_captured = False
            back_file = ''
            stitched_done = False

    if stitched_done and (time.time() - last_stitch_time > cooldown_time):
        captured_flags = {zone: False for zone in captured_flags}
        captured_files = {
            zone: {'left': '', 'top': '', 'right': ''}
            for zone in captured_files
        }
        stitched_done = False
        capture_start_time = None
        back_captured = False
        back_file = ''
        print("[RESET] Siap untuk capture selanjutnya.")

    # Resize semua frame ke ukuran yang sama
    def resize_frame(frame, width=640, height=360):
        return cv2.resize(frame, (width, height))

    frame1_resized = resize_frame(frame_trigger)
    frame2_resized = resize_frame(get_frame(cap2))
    frame3_resized = resize_frame(get_frame(cap3))
    frame4_resized = resize_frame(get_frame(cap4))
    frame5_resized = resize_frame(get_frame(cap5))

    # Gabungkan kamera 1 dan 2 secara horizontal, begitu juga kamera 3 dan 4,5
    top_row = np.hstack((frame1_resized, frame2_resized,
                        frame5_resized))  # Trigger, Left, Back
    bottom_row = np.hstack((frame3_resized, frame4_resized, np.zeros_like(
        frame3_resized)))  # Top, Right, kosong

    # Gabungkan atas dan bawah secara vertikal
    combined_view = np.vstack((top_row, bottom_row))

    # Tampilkan hasil gabungan semua kamera
    cv2.imshow("All Cameras", combined_view)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap_trigger.release()
cap2.release()
cap3.release()
cap4.release()
cap5.release()
cv2.destroyAllWindows()
