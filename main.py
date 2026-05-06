import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('images/left.jpg')
img2 = cv2.imread('images/right.jpg')

if img1 is None or img2 is None:
  print("Images not found:")
else:
  img1_rgb = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
  img2_rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img1_rgb)
plt.title('Left Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_rgb)
plt.title('Right Image')
plt.axis('off')

plt.show()
