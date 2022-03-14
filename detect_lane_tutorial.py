##################################################################################################################
### Adapted from 
# https://github.com/CesarTrevisan/Finding-Lane-Lines-on-the-Road/blob/master/CarND-LaneLines-P1-master/P1.ipynb
###
##################################################################################################################

##################################################################################################################
### Explanation of Hough Transform Lines ###
# https://medium.com/@tomasz.kacmajor/hough-lines-transform-explained-645feda072ab
###
##################################################################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# image = mpimg.imread("solidWhiteCurve.jpg")
# image = mpimg.imread("solidWhiteRight.jpg")
image = mpimg.imread("images/test/my_road_1.jpg")

print("This image is: ", type(image), ' with dimensions: ', image.shape)

plt.imshow(image)
plt.show()

## LANE DETECTION PIPELINE
# 1. Transform image to grayscale
# 2. Apply blur to remove noise
# 3. Use Canny edge detection 
# 4. Define vertices to create a Region of Interest 
# 5. Use Hough Transformation to find lines 
# 6. Merge lines and original image 
# 7. Create pipeline 
# 8. Test with images and videos 

# 1. Transform image to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(grayscale, cmap='gray')
plt.title("Grayscale image")
plt.show()

# 2. Apply blur to remove noise - Gaussian blur
kernel_size = 5
blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
plt.imshow(blur, cmap='gray')
plt.title("Gaussian blur image")
plt.show()

# 3. Use Canny edge detection
low_t = 50
high_t = 150
edges = cv2.Canny(blur, low_t, high_t)
plt.imshow(edges, cmap='gray')
plt.title("Canny edge")
plt.show()

# 4. Define vertices to create a RoI
# Because our camera, suppose we place it on our car dashboard
# It is going to be at a fixed position with little to no movements
# We can create a triangle region that shows only the path in front
vertices = np.array([[(0, 450), (250, 250), (350, 250), (image.shape[1], 450)]], dtype=np.int32)
# Define a mask to start with
mask = np.zeros_like(image)

if len(image.shape) > 2:
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

cv2.fillPoly(mask, vertices, ignore_mask_color)
plt.imshow(mask)
plt.title("Mask")
plt.show()

# Original image with mask
masked_image = cv2.bitwise_and(image, mask)
plt.imshow(masked_image)
plt.title("Masked Image")
plt.show()

# Edges image with mask
mask = np.zeros_like(edges)
if len(edges.shape) > 2:
    channel_count = edges.shape[2]
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
plt.imshow(masked_edges, cmap='gray')
plt.title("Masked Edges")
plt.show()

# 5. Use Hough Transformation to find lines
rho = 3
theta = np.pi / 180
threshold = 15
min_line_len = 150
max_line_gap = 60

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

# Helper function to draw lines
# Aggregates the mean of the lines and 
# Draws the final lines
def draw_lines_mean(img, lines, color=[255, 0, 0], thickness=7):
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []

    y_bottom = 540
    y_upper = 315

    slope = 0
    b = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            # test and filter values to slope
            if ((y2-y1)/(x2-x1)) > 0.5 and ((y2-y1)/(x2-x1)) < 0.8:
                
                slope = ((y2-y1)/(x2-x1))
                b = y1 - slope*x1
                
                x_bottom_pos.append((y_bottom - b)/slope)
                x_upper_pos.append((y_upper - b)/slope)
                                      
            elif ((y2-y1)/(x2-x1)) < -0.5 and ((y2-y1)/(x2-x1)) > -0.8:
            
                slope = ((y2-y1)/(x2-x1))
                b = y1 - slope*x1
                
                x_bottom_neg.append((y_bottom - b)/slope)
                x_upper_neg.append((y_upper - b)/slope)

    # a new 2d array with means 
    lines_mean = np.array([[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper))], 
                            [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper))]])

    # Draw the lines
    for i in range(len(lines_mean)):
        cv2.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)

# Draw all lines found
def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

draw_lines_mean(line_img, lines)
plt.imshow(line_img, cmap='gray')
plt.show()

# 6. Merge lines and original image
lines_edges = cv2.addWeighted(image, 0.8, line_img, 1.0, 0.0)
plt.imshow(lines_edges)
plt.title("Lanes detected")
plt.show()
