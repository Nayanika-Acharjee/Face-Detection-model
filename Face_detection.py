








import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN


image = cv2.imread("sample.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


plt.imshow(image_rgb)
plt.axis("off")
plt.title("Original Image")
plt.show()


detector = MTCNN()
faces = detector.detect_faces(image_rgb)
print("Faces detected:", len(faces))

# Draw boxes + keypoints
for face in faces:
    x, y, w, h = face['box']
    cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0,255,0), 3)

    for key, val in face['keypoints'].items():
        cv2.circle(image_rgb, val, 5, (255,0,0), -1)

# Show result
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Detected Faces")
plt.show()
