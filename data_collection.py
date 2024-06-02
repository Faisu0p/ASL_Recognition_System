import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import csv
import os


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(x, y):
    return math.atan2(y, x) * 180 / math.pi  # Convert radians to degrees


# Directories containing the images
main_dirs = {
    'Alphabets': 'Dataset/ALPHABETS',
    'Numbers': 'Dataset/NUMBERS'
}

detector = HandDetector(maxHands=1)
imgSize = 300

for category, main_dir in main_dirs.items():
    csv_file_path = f'ASL_{category.upper()}.csv'

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Creating the header row for the CSV
        header_row = ['ImageName', 'Class', 'HandType']
        for i in range(21):
            header_row.extend(
                [f'Landmarks_{i}_X', f'Landmarks_{i}_Y', f'Landmarks_{i}_Distance', f'Landmarks_{i}_Angle'])
        csv_writer.writerow(header_row)

        if not os.path.exists(main_dir):
            print(f"The directory {main_dir} does not exist")
            continue

        classes = os.listdir(main_dir)
        for class_name in classes:
            class_dir = os.path.join(main_dir, class_name)

            if not os.path.isdir(class_dir):
                continue  # Skip if not a directory

            image_files = os.listdir(class_dir)
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Could not read image {image_path}. Skipping.")
                    continue  # Skip if the image could not be read

                hands, img = detector.findHands(img)

                if hands:
                    hand = hands[0]
                    handType = hand['type']
                    landmarks = hand['lmList']

                    x, y, w, h = hand['bbox']
                    scaleX = imgSize / w
                    scaleY = imgSize / h
                    scaled_landmarks = [[int(scaleX * point[0]), int(scaleY * point[1])] for point in landmarks]

                    row_data = [image_file, class_name, handType]
                    for i, point in enumerate(scaled_landmarks):
                        x, y = point
                        distance = calculate_distance(0, 0, x, y)
                        angle = calculate_angle(x, y)
                        row_data.extend([x, y, distance, angle])

                    csv_writer.writerow(row_data)
