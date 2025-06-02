import os
import json
import time
import requests
from ultralytics import YOLO
import difflib
import easyocr
import re
import json
import datetime
import cv2
import jwt
from requests.exceptions import ConnectionError


# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load YOLO model with absolute path
yolo_model = YOLO(os.path.join(base_dir, "Weights", "yolov8.pt"))

# definisi class
class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']

# Naik satu folder ke root project
root_dir = os.path.abspath(os.path.join(base_dir, '..'))

# Bangun path akhir
upload_dir = os.path.join(
    root_dir, 'containerdamage', 'monitoring-container-damage-be', 'uploads', 'manual-scan')


def generate_jwt_token():
    # JWT secret from .env (same as in your Node.js app)
    jwt_secret = "cef16827b7a3116309dfadecc811629eb2303056227cfebac15a1c48454ca961"

    # Create payload with expiration time (e.g., 1 hour)
    code = {
        # You can use any identifier that makes sense
        "userId": "container-damage-detector",
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }

    # Generate the token
    token = jwt.encode(code, jwt_secret, algorithm="HS256")
    return token


def ocr_image_to_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image cannot be read: {image_path}")
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

    valid_words = ['22G1', '22G0', '45R1', '22T1']

    # instance text detector
    reader = easyocr.Reader(['en'], gpu=False)
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
    print(combined_text_no_container)
    combined_text_type_container = combined_text[11:15]
    print(combined_text_type_container)

    return combined_text_no_container, combined_text_type_container


def detect_damage_yolo(image_path, prefix):
    results = yolo_model(image_path)[0]
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
        "Lubang": "hole",
        "Patah": "broken",
        "Penyok": "dents",
        "Retak": "crack"
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


def collect_images(upload_dir, image_extensions=(".jpg", ".jpeg", ".png")):
    image_data = {"Back": None, "Left": None, "Top": None, "Right": None}
    container_number = container_type = ""

    for filename in os.listdir(upload_dir):
        if not filename.lower().endswith(image_extensions):
            continue

        prefix = filename.split('-')[0].capitalize()
        if prefix not in image_data:
            continue

        image_path = os.path.join(upload_dir, filename)

        if prefix == "Back":
            container_number, container_type = ocr_image_to_text(image_path)
            if not container_number or len(container_number.strip()) < 12:
                now = datetime.datetime.now()
                default_date_str = now.strftime("%d%m%Y-%H%M%S")
                container_number = f"{default_date_str}-0001"
                container_type = "0000"

        image_data[prefix] = detect_damage_yolo(image_path, prefix)

    return container_number, container_type, image_data


def prepare_payload(container_number, container_type, image_data):
    details_list = []

    for side in ["Back", "Left", "Top", "Right"]:
        if image_data[side]:
            details_list.append(image_data[side]["detail"])

    payload = {
        "no_container": container_number,
        "container_type": container_type,
        "status_container": "IN/OUT",
        "details": json.dumps(details_list)
    }

    print(payload)

    return payload


def prepare_files(image_data):
    files = []
    for side in ["Back", "Left", "Top", "Right"]:
        if image_data[side] and "image_path" in image_data[side]:
            image_path = image_data[side]["image_path"]
            try:
                file = open(image_path, 'rb')
                files.append(
                    ('images', (os.path.basename(image_path), file, 'image/jpeg')))
            except Exception as e:
                print(f"Gagal membuka file {image_path}: {e}")
    return files


def send_to_api(payload, files):
    token = generate_jwt_token()
    url = "http://localhost:3000/api/containers/auto-scan"
    headers = {
        'Authorization': f'Bearer {token}'
    }

    try:
        response = requests.post(url, headers=headers,
                                 data=payload, files=files, timeout=10)
        if response.status_code == 201:
            print("Data berhasil dikirim ke API.")
            print(response.json())
        else:
            print(f"Gagal mengirim data. Status: {response.status_code}")
            print(response.text)
    except ConnectionError as ce:
        print("Gagal koneksi ke API:", ce)
        time.sleep(1.0)
    except Exception as e:
        print("Terjadi error saat mengirim ke API:", e)
    finally:
        for _, file_tuple in files:
            try:
                file_tuple[1].close()
            except Exception as e:
                print("Gagal menutup file:", e)


def cleanup_files(image_data, upload_dir):
    for side in ["Back", "Left", "Top", "Right"]:
        if image_data.get(side):
            if "image_path" in image_data[side]:
                result_path = image_data[side]["image_path"]
                try:
                    if os.path.exists(result_path):
                        os.remove(result_path)
                        print(f"Deleted result: {result_path}")
                        time.sleep(0.2)
                except Exception as e:
                    print(f"Failed to delete result image {result_path}: {e}")

            if "original_path" in image_data[side]:
                input_path = image_data[side]["original_path"]
                try:
                    if os.path.exists(input_path):
                        os.remove(input_path)
                        print(f"Deleted input: {input_path}")
                        time.sleep(0.2)
                except Exception as e:
                    print(f"Failed to delete input image {input_path}: {e}")


def process_scan_manual_images(upload_dir):
    container_number, container_type, image_data = collect_images(upload_dir)

    if not image_data.get("Back"):
        print("Gambar Back harus tersedia.")
        cleanup_files(image_data, upload_dir)
        return

    if not container_number:
        print("Nomor kontainer tidak terdeteksi.")
        cleanup_files(image_data, upload_dir)
        return

    payload = prepare_payload(container_number, container_type, image_data)
    files = prepare_files(image_data)
    send_to_api(payload, files)
    cleanup_files(image_data, upload_dir)


if __name__ == "__main__":
    process_scan_manual_images(upload_dir)
