
import cv2
import jwt
import os
from ultralytics import YOLO
import difflib
import easyocr
import re
import json
import requests
import datetime

# Load YOLO model folder weights
yolo_model = YOLO("Weights/yolov8.pt")
# definisi class
class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']

upload_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # folder tempat script ini
        'monitoring-container-damage-be',
        'uploads',
        'manual-scan'
    )
)


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


def process_scan_manual_images(upload_dir):
    image_extensions = (".jpg", ".jpeg", ".png")
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

        image_data[prefix] = detect_damage_yolo(image_path, prefix)

    if not image_data.get("Back"):
        print("Gambar Back harus tersedia.")
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

    payload = {
        "no_container": container_number,
        "container_type": container_type,
        "details": json.dumps(details_list)
    }

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
    token = generate_jwt_token()
    # print(token)

    url = "http://localhost:3000/api/containers/auto-scan"  # Ganti dengan URL API-mu
    headers = {
        'Authorization': f'Bearer {token}'
    }

    # print(payload)
    # Kirim POST request
    response = requests.post(
        url,
        headers=headers,
        data=payload,
        files=files
    )

    # print(response.text)
    # Tutup semua file setelah upload untuk menghindari resource leak
    for _, file_tuple in files:
        file_tuple[1].close()

    if response.status_code == 201:
        print("Data successfully sent to API")
        print(response.json())

        # Hapus semua file image hasil deteksi
        for side in ["Back", "Left", "Top", "Right"]:
            if image_data[side] and "image_path" in image_data[side]:
                try:
                    os.remove(image_data[side]["image_path"])
                    print(f"Deleted: {image_data[side]['image_path']}")
                except Exception as e:
                    print(
                        f"Failed to delete {image_data[side]['image_path']}: {e}")

        # Hapus juga file gambar input dari folder manual-scan
        for filename in os.listdir(upload_dir):
            if filename.lower().endswith(image_extensions):
                file_path = os.path.join(upload_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted input image: {file_path}")
                except Exception as e:
                    print(f"Failed to delete input image {file_path}: {e}")
    else:
        print(f"Failed to send data. Status code: {response.status_code}")
        print(response.text)
        # Hapus semua file hasil deteksi
        for side in ["Back", "Left", "Top", "Right"]:
            if image_data[side] and "image_path" in image_data[side]:
                try:
                    os.remove(image_data[side]["image_path"])
                    print(f"Deleted: {image_data[side]['image_path']}")
                except Exception as e:
                    print(
                        f"Failed to delete {image_data[side]['image_path']}: {e}")

        # Hapus juga file gambar input dari folder manual-scan
        for filename in os.listdir(upload_dir):
            if filename.lower().endswith(image_extensions):
                file_path = os.path.join(upload_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted input image: {file_path}")
                except Exception as e:
                    print(f"Failed to delete input image {file_path}: {e}")


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

    valid_words = ['22G1', '22G0', '45R1', '22T1', '22UG']

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
    # print(combined_text)
    combined_text_no_container = combined_text[0:11]
    combined_text_type_container = combined_text[11:15]

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


if __name__ == "__main__":
    process_scan_manual_images(upload_dir)
