import cv2
import numpy as np
import joblib
import math
import tkinter as tk
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
import os

# Define the image size for scaling landmarks
imgSize = 300


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(x, y):
    return math.atan2(y, x) * 180 / math.pi


def calculate_landmarks(hand):
    landmarks = hand['lmList']
    x, y, w, h = hand['bbox']
    scaleX = imgSize / w
    scaleY = imgSize / h
    scaled_landmarks = [[int(scaleX * point[0]), int(scaleY * point[1])] for point in landmarks]
    flattened_scaled_landmarks = []
    for point in scaled_landmarks:
        x, y = point
        distance = calculate_distance(0, 0, x, y)
        angle = calculate_angle(x, y)
        flattened_scaled_landmarks.extend([x, y, distance, angle])
    # Add a dummy feature to make the length 85
    flattened_scaled_landmarks.append(0)  # You can change the value as needed
    return flattened_scaled_landmarks


def predict_numbers(hand):
    flattened_scaled_landmarks = calculate_landmarks(hand)
    predictions = number_model.predict([flattened_scaled_landmarks])
    predicted_number = number_class_mapping.get(int(predictions[0]), "Unknown")
    return predicted_number


def predict_alphabets(hand):
    flattened_scaled_landmarks = calculate_landmarks(hand)
    predictions = alphabet_model.predict([flattened_scaled_landmarks])
    predicted_alphabet = alphabet_class_mapping.get(int(predictions[0]), "Unknown")
    return predicted_alphabet


def predict_word(hand):
    flattened_scaled_landmarks = calculate_landmarks(hand)
    predictions = word_model.predict([flattened_scaled_landmarks])
    predicted_word = word_class_mapping.get(int(predictions[0]), "Unknown")
    return predicted_word


def start_detection():
    global stop_flag
    stop_flag = False
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    detection_loop()


def detection_loop():
    global stop_flag
    if stop_flag:
        return
    ret, frame = video_feed.read()
    if not ret:
        label_var.set("Failed to capture video frame")
        return
    hands, _ = detector.findHands(frame)
    for hand in hands:
        if option_var.get() == 1:
            predicted_output = predict_numbers(hand)
        elif option_var.get() == 2:
            predicted_output = predict_alphabets(hand)
        else:
            predicted_output = predict_word(hand)
        label_var.set(f"Predicted: {predicted_output}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)
    if not stop_flag:
        label_img.after(10, detection_loop)


def stop_detection():
    global stop_flag
    stop_flag = True
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)


# Load the saved models
model_files = ['ASL_NUMBERS_MODEL.joblib', 'ASL_ALPHABETS_MODEL.joblib', 'ASL_WORDS.joblib']
if all(os.path.exists(model) for model in model_files):
    number_model = joblib.load('ASL_NUMBERS_MODEL.joblib')
    alphabet_model = joblib.load('ASL_ALPHABETS_MODEL.joblib')
    word_model = joblib.load('ASL_WORDS.joblib')
else:
    raise FileNotFoundError("One or more model files are missing")

# Dictionary to map predicted class to corresponding text for numbers
number_class_mapping = {
    0: "Number Zero",
    1: "Number One",
    2: "Number Two",
    3: "Number Three",
    4: "Number Four",
    5: "Number Five",
    6: "Number Six",
    7: "Number Seven",
    8: "Number Eight",
    9: "Number Nine"
}

# Dictionary to map predicted class to corresponding text for alphabets
alphabet_class_mapping = {
    0: "This Symbol is A",
    1: "This symbol is B",
    2: "This symbol is C",
    3: "This symbol is D",
    4: "This symbol is E",
    5: "This symbol is F",
    6: "This symbol is G",
    7: "This symbol is H",
    8: "This symbol is I",
    9: "This symbol is J",
    10: "This symbol is K",
    11: "This symbol is L",
    12: "This symbol is M",
    13: "This symbol is N",
    14: "This symbol is O",
    15: "This symbol is P",
    16: "This symbol is Q",
    17: "This symbol is R",
    18: "This symbol is S",
    19: "This symbol is T",
    20: "This symbol is U",
    21: "This symbol is V",
    22: "This symbol is W",
    23: "This symbol is X",
    24: "This symbol is Y",
    25: "This symbol is Z",
    26: "This symbol is NOTHING"
}

# Dictionary to map predicted class to corresponding text for words
word_class_mapping = {
    1: "Like",
    2: "Beautiful"
}

# Initializing video capture and hand detector
video_feed = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # Increase maxHands to 2 for detecting both hands

# Initialize Tkinter application
app = tk.Tk()
app.title("ASL Recognition System")

# Set background image
background_image = Image.open('Background_Image.jpg')  # Change the file path to your background image
background_image = background_image.resize((1369, 720), Image.BICUBIC)  # Resize the image as needed
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(app, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a frame for displaying the video feed
frame_container = tk.Frame(app, bd=5, relief=tk.RAISED, highlightbackground="Brown", highlightthickness=5)
frame_container.pack(padx=20, pady=8)

# Create a label for displaying the video feed
label_img = tk.Label(frame_container)
label_img.pack()

# Create a label for displaying the predicted sign or message
label_var = tk.StringVar()
label_var.set("Predicted: None")
label_prediction = tk.Label(app, textvariable=label_var, relief=tk.RAISED, font=('Helvetica', 16))
label_prediction.pack(pady=5)

# Create a variable to store the selected option
option_var = tk.IntVar()
option_var.set(0)

# Create a frame for option buttons
option_frame = tk.Frame(app)
option_frame.pack()

# Create radio buttons for selecting the option
option1_radio = tk.Radiobutton(option_frame, text="Numbers", variable=option_var, value=1, relief=tk.RAISED,
                               borderwidth=3, bg="gray")
option1_radio.pack(side=tk.LEFT, padx=5, pady=5)
option2_radio = tk.Radiobutton(option_frame, text="Alphabets", variable=option_var, value=2, relief=tk.RAISED,
                               borderwidth=3, bg="gray")
option2_radio.pack(side=tk.LEFT, padx=5, pady=5)
option3_radio = tk.Radiobutton(option_frame, text="Words", variable=option_var, value=3, relief=tk.RAISED,
                               borderwidth=3, bg="gray")
option3_radio.pack(side=tk.LEFT, padx=5, pady=5)

# Create a frame for start and stop buttons
button_frame = tk.Frame(app)
button_frame.pack()

# Create buttons to start and stop the sign detection process
start_button = tk.Button(button_frame, text="Start Detection", command=start_detection, bg="green", fg="white",
                         relief=tk.RAISED, borderwidth=3)
stop_button = tk.Button(button_frame, text="Stop Detection", command=stop_detection, state=tk.DISABLED, bg="red",
                        fg="white", relief=tk.RAISED, borderwidth=3)

# Pack buttons in a horizontal line
start_button.pack(side=tk.LEFT, padx=5, pady=5)
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

# Variable to control the stop flag
stop_flag = False

# Start the Tkinter event loop
app.mainloop()

# Release the video feed when the application closes
video_feed.release()
