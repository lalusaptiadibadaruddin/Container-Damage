import cv2
import math
import cvzone
import requests
import os
import jwt
import datetime
import time
from ultralytics import YOLO

# Load YOLO model folder weights
yolo_model = YOLO("Weights/yolov8.pt")

# yolo_model = YOLO("Weights/yolov8image.pt")

# definisi class
class_labels = ['Karat', 'Lubang', 'Patah', 'Penyok', 'Retak']
# class_labels = ['Dent', 'Hole', 'Rust']
# print(class_labels)

# JWT token generation function
def generate_jwt_token():
    # JWT secret from .env (same as in your Node.js app)
    jwt_secret = "cef16827b7a3116309dfadecc811629eb2303056227cfebac15a1c48454ca961"
    
    # Create payload with expiration time (e.g., 1 hour)
    payload = {
        "userId": "container-damage-detector",  # You can use any identifier that makes sense
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    
    # Generate the token
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return token

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

# Create temp folder if it doesn't exist
temp_folder = "temp_files"
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

# Generate a unique filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
temp_filename = f"container_damage_{timestamp}.jpg"
output_image_path = os.path.join(temp_folder, temp_filename)

# Save the processed image temporarily
cv2.imwrite(output_image_path, img)

# Prepare data to send to API
# Map Indonesian labels to English for the API
label_mapping = {
    'Karat': 'rust',
    'Lubang': 'holes',
    'Patah': 'breaks',
    'Penyok': 'dents',
    'Retak': 'cracks'
}

# Prepare form data and file handle properly
file_handle = open(output_image_path, 'rb')
files = {
    'image': (temp_filename, file_handle, 'image/jpeg')
}

form_data = {
    'totalDamage': str(count),
    'rust': str(damage_counts['Karat']),
    'holes': str(damage_counts['Lubang']),
    'breaks': str(damage_counts['Patah']),
    'dents': str(damage_counts['Penyok']),
    'cracks': str(damage_counts['Retak'])
}

# Generate JWT token
token = generate_jwt_token()

# Send data to API
try:
    api_url = "http://localhost:5000/api/container-damage/"
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    response = requests.post(api_url, files=files, data=form_data, headers=headers)
    
    if response.status_code == 201:
        print("Data successfully sent to API")
        print(response.json())
    else:
        print(f"Failed to send data. Status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error sending data to API: {str(e)}")
finally:
    # Close the file handle first
    file_handle.close()
    
    # Wait a moment to ensure file is fully released by the OS
    time.sleep(0.5)
    
    # Now try to remove the file
    try:
        if os.path.exists(output_image_path):
            os.remove(output_image_path)
            print(f"Temporary file {output_image_path} deleted")
    except Exception as e:
        print(f"Warning: Could not delete temporary file: {str(e)}")
        print(f"Temporary file location: {output_image_path}")

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
