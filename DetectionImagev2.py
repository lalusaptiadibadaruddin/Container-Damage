import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import re
import difflib

# read image
image_path = 'container.jpg'
img = cv2.imread(image_path)

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

    if score > threshold:
        # Adjust bounding box coordinates to match the original image
        # Need to scale back down and offset for the right half
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
                      (int(bbox_adjusted[0][0]), int(bbox_adjusted[0][1])),
                      (int(bbox_adjusted[2][0]), int(bbox_adjusted[2][1])),
                      (0, 0, 255), 2)
        cv2.putText(vis_img, cleaned_text,
                    (int(bbox_adjusted[0][0]), int(bbox_adjusted[0][1]) - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 0, 0), 2)

# Only combine the first 3 lines, keep the rest separate
first_three = [text for i, text, _ in all_detected_text[:3] if text]
combined_text = ''.join(first_three)
print("Container Number:", combined_text)

# Print the rest individually
if len(all_detected_text) > 3:
    print("Other detected text:")
    for i, text, _ in all_detected_text[3:]:
        if text:
            print(f"  {text}")

# Display the image with the OCR results
plt.figure(figsize=(12, 10))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.title("OCR Detection (Zoomed Right Side)")
plt.show()
