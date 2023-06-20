import cv2
import numpy as np

# Load image
#image = cv2.imread(r"C:\Users\Jose Gil\PycharmProjects\PRUEBA2OK\1239x697IMG.jpg")
image = cv2.imread(r"C:\Users\ASDJF\OneDrive\Escritorio\PROYECT VISUAL IA\pruebasia\Mask_RCNN\dataset2\images\image6.png")
#image = cv2.imread(r"c:\Users\ASDJF\OneDrive\Escritorio\PROYECT VISUAL IA\DETECCION_AVANZADA\DETECCION_POR_COLOR\ara_flores.png")

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range of color in HSV
#lower = np.array([85, 0, 0])
#upper = np.array([147, 255, 255])
#lower = np.array([0, 169, 0])
#upper = np.array([60, 255, 255])
lower = np.array([0, 155, 0])
upper = np.array([63, 255, 114])


# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower, upper)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image, image, mask=mask)

# Apply Canny Edge Detection
edges = cv2.Canny(res, 50, 150)

# Resize images
resized_image = cv2.resize(image, (800, 500))
resized_mask = cv2.resize(mask, (800, 500))
resized_res = cv2.resize(res, (800, 500))
#resized_edges = cv2.resize(edges, (800, 500))

#cv2.imshow('frame', resized_image)
cv2.imshow('mask', resized_mask)
cv2.imshow('res', resized_res)
#cv2.imshow('edges', resized_edges)

# Ruta y nombre del archivo de salida
#output_path = r"c:\Users\ASDJF\OneDrive\Escritorio\PROYECT VISUAL IA\DETECCION_AVANZADA\DETECCION_POR_COLOR\ara_floresres2.png"

# Guardar la máscara en la ruta especificada
#cv2.imwrite(output_path, resized_res)

cv2.waitKey(0)
cv2.destroyAllWindows()
