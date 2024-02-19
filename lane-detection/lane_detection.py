import cv2
import numpy as np
import time

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold=50, high_threshold=200):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def slope_lines(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue 
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            if -1e6 < m < 1e6 and m != 0:  
                if m < 0:
                    left_lines.append((m, c))
                elif m >= 0:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0) if left_lines else None
    right_line = np.mean(right_lines, axis=0) if right_lines else None

    if left_line is not None:
        slope, intercept = left_line
        rows, cols = image.shape[:2]
        y1 = int(rows)
        y2 = int(rows * 0.6)
        if slope != 0:  
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            poly_vertices.append((x1, y1))
            poly_vertices.append((x2, y2))

    if right_line is not None:
        slope, intercept = right_line
        rows, cols = image.shape[:2]
        y1 = int(rows)
        y2 = int(rows * 0.6)
        if slope != 0: 
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            poly_vertices.append((x1, y1))
            poly_vertices.append((x2, y2))

    if len(poly_vertices) == 4:
        poly_vertices = [poly_vertices[i] for i in order]
        cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(0, 170, 120))

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    line_img = slope_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.1, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.15, rows * 0.9]
    top_left = [cols * 0.45, rows * 0.58]
    bottom_right = [cols * 0.95, rows * 0.9]
    top_right = [cols * 0.55, rows * 0.58]
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

def detect_lines(image):
    gray_img = grayscale(image)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)
    masked_img = region_of_interest(img=canny_img, vertices=get_vertices(image))
    houghed_lines = hough_lines(img=masked_img, rho=1, theta=np.pi / 180, threshold=20, min_line_len=20, max_line_gap=180)
    output = weighted_img(img=houghed_lines, initial_img=image, a=0.8, b=1., c=0.)
    return output

def detect_lanes(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        output_frame = detect_lines(frame)

        out.write(output_frame)

        cv2.imshow('Lane Detection', output_frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_video_path = 'test_1.mp4'
output_video_path = 'output1.mp4'

detect_lanes(input_video_path, output_video_path)