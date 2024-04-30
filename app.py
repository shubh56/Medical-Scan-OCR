import easyocr
from matplotlib import pyplot as plt
import cv2
import numpy as np


image_path = 'testReport.png'
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(image_path)
res = list(x[1] for x in result)
print(res)
for a in res:
    with open("texts", mode='a') as file:
        file.write(a + "\n")

# top_left = tuple(result[0][0][0])
# bottom_right = tuple(result[0][0][2])
# text = result[0][1]
# font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread(image_path)
# img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
# img = cv2.putText(img, text, top_left, font, .5, (255, 255, 255), cv2.LINE_AA)
# plt.imshow(img)
# plt.show()
for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
    # img = cv2.putText(img, text, top_left, font, 0.5, (0, 0, 0), cv2.LINE_AA)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()