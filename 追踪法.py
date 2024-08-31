import cv2
import numpy as np
import cv2
import numpy as np
from tkinter import *
from math import *
from tkinter.filedialog import *

import numpy as np
import scipy
from PIL import ImageTk, Image
import cv2
from scipy import ndimage
from scipy import signal
def choosePic():
    global fingerprint_image
    img_path = askopenfilename(initialdir='./', title='选择待识别图片',
                               filetypes=[("tif", "*.tif"), ("jpg", "*.jpg"), ("png", "*.png")])
    if img_path:
        print(img_path)
        fingerprint_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # print(type(fingerprint_image))
        if len(fingerprint_image.shape) > 2:
            fingerprint_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)  # 转灰度图

        rows, cols = np.shape(fingerprint_image)
        aspect_ratio = np.double(rows) / np.double(cols)

        new_rows = 320  # randomly selected number
        new_cols = new_rows / aspect_ratio

        fingerprint_image = cv2.resize(fingerprint_image, (int(new_cols), int(new_rows)))
    root.destroy()


# 指纹图像归一化
def normalize_fingerprint(fingerprint_image):
    M = np.mean(fingerprint_image)
    VAR = np.var(fingerprint_image)
    M_0 = 140
    VAR_0 = 600
    normalized_image = np.zeros_like(fingerprint_image)
    for i in range(fingerprint_image.shape[0]):
        for j in range(fingerprint_image.shape[1]):
            if fingerprint_image[i, j] >= M:
                normalized_image[i, j] = M_0 + np.sqrt(VAR_0 * (fingerprint_image[i, j] - M) ** 2 / VAR)
            else:
                normalized_image[i, j] = M_0 - np.sqrt(VAR_0 * (fingerprint_image[i, j] - M) ** 2 / VAR)
    return normalized_image

# 前景与背景分离
def segment_fingerprint(normalized_image, block_size=16, threshold=50):
    height, width = normalized_image.shape
    segmented_image = np.zeros_like(normalized_image, dtype=np.uint8)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = normalized_image[i:i+block_size, j:j+block_size]
            mean = np.mean(block)
            variance = np.var(block)
            
            if variance >= threshold:
                segmented_image[i:i+block_size, j:j+block_size] = block
    
    return segmented_image

# 计算梯度和方向信息
def compute_gradients(normalized_image):
    gradient_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3).astype(np.float32)
    gradient_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3).astype(np.float32)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2).astype(np.float32)
    gradient_angle = np.arctan2(gradient_y, gradient_x) + np.pi/2
    return gradient_magnitude, gradient_angle

# 平滑方向信息
def smooth_directions(directions, block_size=5):
    kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size**2)
    smoothed_directions = cv2.filter2D(directions, -1, kernel)
    return smoothed_directions

# 计算方向信息的可靠性
def compute_reliability(directions, block_size=5):
    height, width = directions.shape
    reliability = np.zeros_like(directions, dtype=np.float32)
    
    for i in range(block_size//2, height-block_size//2):
        for j in range(block_size//2, width-block_size//2):
            block = directions[i-block_size//2:i+block_size//2+1, j-block_size//2:j+block_size//2+1]
            diff = np.abs(block - directions[i, j])
            reliability[i, j] = np.sqrt(np.sum(diff**2)) / block_size
    
    return reliability

# 滤波增强邻域点
def enhance_neighborhood_points(neighbor_points, image):
    filtered_points = []
    
    for point in neighbor_points:
        i, j = point
        plane1 = image[max(i-1, 0):min(i+2, image.shape[0]), max(j-1, 0):min(j+2, image.shape[1])]
        plane2 = image[max(i-2, 0):min(i+3, image.shape[0]), max(j-2, 0):min(j+3, image.shape[1])]
        plane3 = image[max(i-3, 0):min(i+4, image.shape[0]), max(j-3, 0):min(j+4, image.shape[1])]
        
        # 检查邻域是否为空
        if plane1.size > 0 and plane2.size > 0 and plane3.size > 0:
            # 平滑处理
            smoothed_point = np.zeros_like(plane1, dtype=np.float32)
            count = 0
            
            if not np.isnan(plane1).all():
                smoothed_point += np.nanmean(plane1)
                count += 1
            if not np.isnan(plane2).all():
                smoothed_point += np.nanmean(plane2)
                count += 1
            if not np.isnan(plane3).all():
                smoothed_point += np.nanmean(plane3)
                count += 1
            
            # 特殊处理邻域大小不足的情况
            if count == 0:
                smoothed_point = image[i, j]
                count = 1
            
            # 高斯滤波
            filtered_value = np.nanmean(smoothed_point) / count
            filtered_points.append(filtered_value)
    
    return filtered_points

# 计算灰度之差
def compute_gray_difference(center_point, next_point, image):
    center_x, center_y = center_point
    next_x, next_y = next_point
    center_value = image[center_y, center_x]
    next_value = image[next_y, next_x]
    
    if np.isnan(center_value) or np.isnan(next_value):
        return 0
    
    return center_value - next_value

# 角度调整函数
def adjust_angle(curr_direction, next_direction, max_point, curr_point):
    beta = np.arctan2(max_point[1] - curr_point[1], max_point[0] - curr_point[0])
    adjusted_direction = next_direction
    
    if curr_direction >= 0 and curr_direction < np.pi/2 and next_direction >= np.pi/2 and next_direction < np.pi:
        if beta >= 0 and beta <= np.pi/6:
            adjusted_direction += np.pi
    elif curr_direction >= np.pi/2 and curr_direction < np.pi and next_direction >= 0 and next_direction < np.pi/2:
        if beta >= 0 and beta <= np.pi/6:
            adjusted_direction += np.pi
    elif curr_direction >= np.pi and curr_direction < 3 * np.pi/2 and next_direction >= 3 * np.pi/2 and next_direction < 2 * np.pi:
        if beta >= 0 and beta <= np.pi/6:
            adjusted_direction -= np.pi
    elif curr_direction >= 3 * np.pi/2 and curr_direction < 2 * np.pi and next_direction >= np.pi and next_direction < 3 * np.pi/2:
        if beta >= 0 and beta <= np.pi/6:
            adjusted_direction -= np.pi
    
    return adjusted_direction

# 结束条件判断函数
def check_termination_condition(curr_point, next_point, gray_difference_threshold):
    gray_difference = compute_gray_difference(curr_point, next_point, normalized_image)
    return abs(gray_difference) < gray_difference_threshold

# 计算邻域点集
def compute_neighborhood_points(center_x, center_y, curr_direction):
    sigma = 7
    neighbor_points = []
    
    if curr_direction >= 0 and curr_direction < np.pi/2:
        for j in range(center_x - int(sigma * np.cos(curr_direction)), center_x + int(sigma * np.cos(curr_direction)) + 1):
            i = center_y + int((j - center_x) * np.tan(curr_direction))
            neighbor_points.append((i, j))
    elif curr_direction >= np.pi/2 and curr_direction < np.pi:
        curr_direction -= np.pi
        for j in range(center_x - int(sigma * np.cos(curr_direction)), center_x + int(sigma * np.cos(curr_direction)) + 1):
            i = center_y - int((j - center_x) * np.tan(curr_direction))
            neighbor_points.append((i, j))
    elif curr_direction >= np.pi and curr_direction < 3 * np.pi/2:
        curr_direction -= np.pi
        for j in range(center_x - int(sigma * np.cos(curr_direction)), center_x + int(sigma * np.cos(curr_direction)) + 1):
            i = center_y + int((j - center_x) * np.tan(curr_direction))
            neighbor_points.append((i, j))
    else:
        curr_direction -= 2 * np.pi
        for j in range(center_x - int(sigma * np.cos(curr_direction)), center_x + int(sigma * np.cos(curr_direction)) + 1):
            i = center_y - int((j - center_x) * np.tan(curr_direction))
            neighbor_points.append((i, j))
    
    return neighbor_points

# 追踪纹线
def trace_ridges(start_x, start_y, direction):
    ridge_points = []
    current_x = start_x
    current_y = start_y
    
    while True:
        ridge_points.append((current_y, current_x))
        
        # 计算下一个跟踪点的坐标
        next_x = int(current_x + step_length * np.cos(direction))
        next_y = int(current_y + step_length * np.sin(direction))
        
        # 检查下一个跟踪点是否在图像范围内
        if next_x < 0 or next_x >= normalized_image.shape[1] or next_y < 0 or next_y >= normalized_image.shape[0]:
            break
        
        # 检查下一个跟踪点是否已被访问过
        if (next_x, next_y) in ridge_points:
            break
        
        # 更新当前点的坐标和方向
        current_x = next_x
        current_y = next_y
        current_direction = smoothed_angles[current_y, current_x]
        
        # 检查跟踪点方向是否与起始方向相差超过阈值
        if abs(current_direction - direction) > direction_threshold:
            break
        
        # 对邻域点集作局部滤波增强处理
        neighbor_points = compute_neighborhood_points(current_x, current_y, current_direction)
        
        if not neighbor_points:
            # 邻域大小不足，跳过当前点，继续寻找下一个点
            break
        
        enhanced_points = enhance_neighborhood_points(neighbor_points, normalized_image)
        
        # 计算下一个跟踪点与当前点的灰度之差
        gray_difference = compute_gray_difference((current_x, current_y), (next_x, next_y), normalized_image)
        
        if gray_difference > gray_difference_threshold:
            # 在滤波处理后的邻域点集中定位最大灰度点和最小灰度点
            max_index = np.argmax(enhanced_points)
            max_point = neighbor_points[max_index]
            
            # 对角度进行调整
            adjusted_direction = adjust_angle(current_direction, smoothed_angles[max_point[0], max_point[1]], max_point, (current_x, current_y))
            
            # 检查结束条件
            if check_termination_condition((current_x, current_y), max_point, gray_difference_threshold):
                break
            
            # 更新当前点的坐标和方向
            current_x = max_point[1]
            current_y = max_point[0]
            direction = adjusted_direction
        else:
            # 检查结束条件
            if check_termination_condition((current_x, current_y), (next_x, next_y), gray_difference_threshold):
                break
            
            # 更新当前点的坐标
            current_x = next_x
            current_y = next_y
    
    return ridge_points


if __name__ == "__main__":
    global fingerprint_image
    root = Tk()
    root.title("脊线特征处理")
    root.geometry('200x200')

    btnChoose = Button(root, text="选择图片", command=choosePic)
    btnChoose.place(x=70, y=70)
    root.mainloop()
    # 加载指纹图像
    # fingerprint_image = cv2.imread('./DB3_B/DB3_B/101_1.tif', cv2.IMREAD_GRAYSCALE)

    # 第一步：显示原始的灰度指纹图像
    cv2.imshow('Original Grayscale Fingerprint Image', fingerprint_image)
    # cv2.waitKey(0)

    # 第二步：指纹图像归一化
    normalized_image = normalize_fingerprint(fingerprint_image).astype(np.uint8)
    cv2.imshow('Normalized Fingerprint Image', normalized_image)
    # cv2.waitKey(0)

    # 第三步：前景与背景分离
    segmented_image = segment_fingerprint(normalized_image)
    cv2.imshow('Segmented Fingerprint Image', segmented_image)
    # cv2.waitKey(0)

    # 第四步：计算梯度和方向信息
    gradient_magnitude, gradient_angle = compute_gradients(normalized_image)

    # 第五步：平滑方向信息
    smoothed_angles = smooth_directions(gradient_angle)

    # 第六步：跟踪纹线
    start_x, start_y, direction = 200, 200, 0
    step_length = 1
    direction_threshold = np.pi/4
    gray_difference_threshold = 8
    ridge_points = trace_ridges(start_x, start_y, direction)
    ridge_image = np.zeros_like(normalized_image, dtype=np.uint8)  # Corrected initialization
    for point in ridge_points:
        ridge_image[point[0], point[1]] = 255  # Set the pixels to white (255)
    cv2.imshow('ridge_image',ridge_image)
    # 在 Smoothed Angles 图像上添加特征点
    for point in ridge_points:
        y, x = point
        cv2.circle(smoothed_angles, (x, y), 2, 255, -1)

    cv2.imshow('Smoothed Angles with Feature Points', (smoothed_angles / np.pi * 255).astype(np.uint8))
    # cv2.waitKey(0)

    # 第七步：提取细节特征
    reliability_threshold = 0.5
    reliabilities = compute_reliability(smoothed_angles)
    filtered_image = normalized_image.copy()
    filtered_image[reliabilities < reliability_threshold] = 0
    cv2.imshow('Detail Feature Extraction Result', filtered_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
