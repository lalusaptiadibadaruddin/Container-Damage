
import cv2
import easyocr
import matplotlib.pyplot as plt

# read image
image_path = 'container.jpg'
img = cv2.imread(image_path)

#instance  text detector
reader = easyocr.Reader(['en'], gpu=False)

#detect text on image
text_ = reader.readtext(img)
threshold = 0.25
#draw text on image
for t in text_:
    # print(t)
    bbox, text , score  = t
    print(text)
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0,255,0),5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.50, (255,0,0), 2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
