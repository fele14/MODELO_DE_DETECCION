import cv2
import numpy as np

# Load image
image = cv2.imread(r"C:\Users\ASDJF\OneDrive\Escritorio\PROYECT VISUAL IA\pruebasia\Mask_RCNN\dataset2\images\image6.png")

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range of color in HSV
lower = np.array([0, 155, 0])
upper = np.array([63, 255, 114])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower, upper)

# Find contours of white points
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255, 50), 2)  # Red rectangle with transparency
    
    # Check top side of the bounding box
    for i in range(x, x + w):
        if y > 0 and mask[y - 1, i] == 0 and y > 1 and mask[y - 2, i] == 0 and y > 2 and mask[y - 3, i] == 0:  # Check if the pixels above are black
            cv2.circle(image, (i, y - 1), 3, (255, 0, 0), -1)  # Blue circle

    # Check bottom side of the bounding box
    for i in range(x, x + w):
        if y + h < mask.shape[0] - 1 and mask[y + h, i] == 0 and y + h < mask.shape[0] - 2 and mask[y + h + 1, i] == 0 and y + h < mask.shape[0] - 3 and mask[y + h + 2, i] == 0:  # Check if the pixels below are black
            cv2.circle(image, (i, y + h), 3, (255, 0, 0), -1)  # Blue circle

# Resize images
resized_image = cv2.resize(image, (800, 500))
resized_mask = cv2.resize(mask, (800, 500))

cv2.imshow('image', resized_image)
cv2.imshow('mask', resized_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
