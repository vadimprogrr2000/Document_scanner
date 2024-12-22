# импортирование необходимых библиотек
import numpy as np
import cv2
import imutils
# параметр для сканируемого изображения
args_image = "fuel_bill_to_scan.jpg"
# прочитать изображение
image = cv2.imread(args_image)
orig = image.copy()
# показать исходное изображение
cv2.imshow("Original Image", image)
cv2.waitKey(0) # press 0 to close all cv2 windows
cv2.destroyAllWindows()