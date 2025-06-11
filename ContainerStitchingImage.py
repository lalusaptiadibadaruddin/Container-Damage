import cv2
import os
import numpy as np
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, '..'))
upload_dir = os.path.join(
    root_dir, 'monitoring-container-damage-be', 'uploads', 'manual-scan')


def stitch_images(upload_dir, label):
    # Ambil 3 file gambar dari folder
    image_files = sorted([f for f in os.listdir(
        upload_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:3]

    if len(image_files) < 3:
        print("Tidak ditemukan 3 gambar.")
        return

    # Baca semua gambar
    raw_images = []
    for file in image_files:
        img_path = os.path.join(upload_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Gagal membaca {img_path}")
            return
        raw_images.append(img)

    # Cari tinggi terkecil
    min_height = min(img.shape[0] for img in raw_images)

    # Resize semua gambar ke tinggi terkecil
    resized_images = []
    for img in raw_images:
        scale = min_height / img.shape[0]
        resized_img = cv2.resize(
            img, (int(img.shape[1] * scale), min_height),
            interpolation=cv2.INTER_AREA
        )
        resized_images.append(resized_img)

    # Gabungkan secara horizontal
    result = np.hstack(resized_images)

    # Buat nama file hasil: {label}-{timestamp}.jpg
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{label}-{timestamp}.jpg"
    output_path = os.path.join(upload_dir, filename)

    # Simpan gambar hasil
    cv2.imwrite(output_path, result)
    print(f"Gambar gabungan disimpan di: {output_path}")


if __name__ == "__main__":
    label = "left"
    stitch_images(upload_dir, label)
