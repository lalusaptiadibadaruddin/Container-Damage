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
import sys
from requests.exceptions import ConnectionError


# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load YOLO model with absolute path
yolo_model = YOLO(os.path.join(base_dir, "Weights", "yolov8.pt"))

# definisi class
class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']


def generate_upload_path(scan_type="manual", container_uid=None):
    root_dir = os.path.abspath(os.path.join(base_dir, '..'))
    # now = datetime.datetime.now()
    # timestamp = now.strftime("%d%m%Y-%H%M%S")

    if scan_type == "manual":
        upload_dir = os.path.join(
            root_dir, 'monitoring-container-damage-be', 'uploads', 'manual-scan')
    elif scan_type == "rescan":
        if container_uid:
            # For rescan, we need to find the original folder for this container
            # We'll search in the container-damages directory structure
            container_damages_dir = os.path.join(
                root_dir, 'monitoring-container-damage-be', 'uploads', 'container-damages')

            # Search for the container's original folder
            upload_dir = None
            if os.path.exists(container_damages_dir):
                for container_folder in os.listdir(container_damages_dir):
                    container_path = os.path.join(container_damages_dir, container_folder)
                    if os.path.isdir(container_path):
                        for date_folder in os.listdir(container_path):
                            date_path = os.path.join(container_path, date_folder)
                            if os.path.isdir(date_path):
                                original_path = os.path.join(date_path, 'original', container_uid)
                                if os.path.exists(original_path):
                                    upload_dir = original_path
                                    break
                        if upload_dir:
                            break

            if not upload_dir:
                raise ValueError(f"Original images folder not found for container_uid: {container_uid}")
        else:
            raise ValueError("container_uid is required for rescan")
    else:
        raise ValueError("Invalid scan_type. Use 'manual' or 'rescan'.")

    # For manual scan, create directory if it doesn't exist
    if scan_type == "manual":
        os.makedirs(upload_dir, exist_ok=True)

    return upload_dir


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

    valid_words = ['22G1', '22G0', '45R1', '22T1', '42G1', '45G1']

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
    # print(combined_text_no_container)
    combined_text_type_container = combined_text[11:15]
    # print(combined_text_type_container)

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
            if not container_number or len(container_number.strip()) != 11:
                now = datetime.datetime.now()
                default_date_str = now.strftime("%d%m%Y-%H%M%S")
                container_number = f"{default_date_str}-0001"
                container_type = "0000"

        image_data[prefix] = detect_damage_yolo(image_path, prefix)

    return container_number, container_type, image_data


def check_image_brightness(image_path, threshold=100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "gelap"
    mean_brightness = img.mean()
    return "terang" if mean_brightness > threshold else "gelap"


def prepare_payload(container_number, container_type, image_data, status_container="IN", user_id=None, container_uid=None, scan_type="manual"):
    details_list = []

    # Check each side status (hasil ok jika image_data[side] ada dan categories detected)
    side_status = {}
    image_status = {}
    # for side in ["Back", "Left", "Top", "Right"]:
    #     if image_data.get(side):
    #         categories = image_data[side]["detail"].get("categories", [])
    #         # Anggap 'hasil ok' jika total damage lebih dari 0 untuk kategori apa pun
    #         total_damage = sum(int(c.get("damageTotal", 0))
    #                            for c in categories)
    #         side_status[side.lower()] = "damage" if total_damage > 0 else "ok"
    #     else:
    #         side_status[side.lower()] = "ok"

    # Cek brightness tiap sisi

    # for side in ["Back", "Left", "Top", "Right"]:
    #     if image_data.get(side) and "image_path" in image_data[side]:
    #         image_status[side.lower()] = check_image_brightness(
    #             image_data[side]["image_path"])
    #     else:
    #         # default jika gambar tidak tersedia
    #         image_status[side.lower()] = "gelap"

    # Status nomor kontainer
    no_container_status = "terdeteksi" if container_number and len(
        container_number.strip()) >= 11 else "tidak terdeteksi"

    # Track condition status for each side to determine overall status
    condition_statuses = []

    for side in ["Back", "Left", "Top", "Right"]:
        side_lower = side.lower()
        if image_data[side]:
            # Ambil path gambar dari image_data
            image_path = image_data[side].get("image_path")
            if image_path:
                # Panggil fungsi check_image_brightness untuk cek terang/gelap
                image_status[side_lower] = check_image_brightness(image_path)
                # Set side_status berdasarkan kualitas gambar, bukan damage
                side_status[side_lower] = "ok" if image_status[side_lower] == "terang" else "not ok"
            else:
                image_status[side_lower] = "normal"
                side_status[side_lower] = "not ok"  # Jika tidak ada gambar, status not ok

            # Set condition_status based on side_status
            condition_status = "success" if side_status[side_lower] == "ok" else "failed"
            condition_statuses.append(condition_status)

            # Gabungkan semua detail + status ke dalam details_list
            detail = image_data[side]["detail"].copy()
            detail.update({
                "side_status": side_status[side_lower],
                "image_status": image_status[side_lower],
                "condition_status": condition_status
            })
            details_list.append(detail)
        else:
            side_status[side_lower] = "not ok"  # Jika tidak ada gambar, status not ok
            image_status[side_lower] = "normal"
            condition_statuses.append("failed")  # Default to failed for missing sides

    # Overall status is success only if all condition_statuses are success
    all_conditions_success = all(cs == "success" for cs in condition_statuses)
    status = "success" if all_conditions_success else "failed"

    payload = {
        "no_container": container_number,
        "container_type": container_type,
        "status_container": status_container,
        "status": status,
        "no_container_status": no_container_status,
        "details": json.dumps(details_list),
        "scan_type": scan_type
    }

    # Add optional parameters if provided
    if user_id is not None:
        payload["userId"] = user_id

    # container_uid is only required for OUT status
    if status_container == "OUT":
        if container_uid is None:
            raise ValueError(
                "container_uid is required when status_container is 'OUT'")
        payload["container_uid"] = container_uid
    elif status_container == "IN" and container_uid is not None:
        # For IN status, container_uid is optional but can be included if provided
        payload["container_uid"] = container_uid

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
    url = "http://localhost:5000/api/containers/auto-scan"
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
                # Hanya hapus file yang mengandung "_detected"
                if "_detected" in result_path:
                    try:
                        if os.path.exists(result_path):
                            os.remove(result_path)
                            print(f"Deleted detected result: {result_path}")
                            time.sleep(0.2)
                    except Exception as e:
                        print(f"Failed to delete detected result image {result_path}: {e}")

            if "original_path" in image_data[side]:
                input_path = image_data[side]["original_path"]
                # Hanya hapus file yang mengandung "_detected"
                if "_detected" in input_path:
                    try:
                        if os.path.exists(input_path):
                            os.remove(input_path)
                            print(f"Deleted detected input: {input_path}")
                            time.sleep(0.2)
                    except Exception as e:
                        print(f"Failed to delete detected input image {input_path}: {e}")


def process_scan_manual_images(scan_type="manual", status_container="IN", user_id=None, container_uid=None):
    upload_dir = generate_upload_path(scan_type=scan_type, container_uid=container_uid)
    container_number, container_type, image_data = collect_images(upload_dir)

    if not image_data.get("Back"):
        print("Gambar Back harus tersedia.")
        print(upload_dir)
        cleanup_files(image_data, upload_dir)
        return

    if not container_number:
        print("Nomor kontainer tidak terdeteksi.")
        cleanup_files(image_data, upload_dir)
        return

    try:
        payload = prepare_payload(container_number, container_type,
                                  image_data, status_container, user_id, container_uid, scan_type)
        files = prepare_files(image_data)
        send_to_api(payload, files)
        print(payload)
    except ValueError as e:
        print(f"Error: {e}")
        cleanup_files(image_data, upload_dir)
        return

    cleanup_files(image_data, upload_dir)


if __name__ == "__main__":
    # Parse command line arguments
    scan_type = "manual"
    status_container = "IN"
    user_id = None
    container_uid = None

    # Check if parameters were passed from Node.js
    if len(sys.argv) > 1:
        try:
            # Parse JSON data from command line argument
            script_data = json.loads(sys.argv[1])
            status_container = script_data.get("status_container", "IN")
            scan_type = script_data.get("scan_type")
            user_id = script_data.get("userId")
            container_uid = script_data.get("container_uid")

            print(
                f"Received parameters: status_container={status_container}, userId={user_id}, container_uid={container_uid}")
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing command line arguments: {e}")
            print("Using default parameters")

    process_scan_manual_images(
        scan_type, status_container, user_id, container_uid)