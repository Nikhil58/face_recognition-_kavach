import cv2

if hasattr(cv2, 'face'):
    print('cv2.face is installed.')
else:
    print('cv2.face is not installed.')
import cv2

# Load grayscale image
gray_image = cv2.imread('user1.jpg', cv2.IMREAD_GRAYSCALE)

# Convert grayscale image to RGB color space
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# Display RGB image
cv2.imshow('RGB image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
