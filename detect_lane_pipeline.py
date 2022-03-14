import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

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
def draw_lines(img, lines, color=[0, 0, 255], thickness=7):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def detect(frame, image_path=None):
    if image_path is not None:
        image = mpimg.imread(image_path)

    else:
        image = frame

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)

    vertices = np.array([[(0, 450), (250, 250), (350, 250), (image.shape[1], 450)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    if len(edges.shape) > 2:
        ignore_mask_color = (255,) * edges.shape[2]
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_edges = cv2.bitwise_and(edges, mask)

    rho = 3
    theta = np.pi/180
    threshold = 15
    min_line_len = 150
    max_line_gap = 50

    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros_like(image)

    draw_lines(line_img, lines)

    alpha = 0.8
    beta = 1.0
    gamma = 0.0
    lines_edges = cv2.addWeighted(image, alpha, line_img, beta, gamma)

    return lines_edges