import cv2
import mediapipe as mp
import time
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid PyInstaller issues
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
import pygame  # Replaced playsound for better reliability
import joblib
import sys
import os

# Initialize pygame mixer for audio
pygame.mixer.init()

# Resource path handling for PyInstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load classifier with error handling
try:
    model = joblib.load(resource_path("flexion_extension_classifier.pkl"))
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Constants
FLEXION_CLASS = 0
EXTENSION_CLASS = 1

# Audio files (ensure these exist in your project directory)
AUDIO_FILES = {
    "left": resource_path("left_hand_detected.mp3"),
    "right": resource_path("right_hand_detected.mp3"),
    "fist": resource_path("fist_held.mp3"),
    "extension": resource_path("extension_done.mp3")
}

# Play audio safely
def play_audio(audio_type):
    try:
        sound = pygame.mixer.Sound(AUDIO_FILES[audio_type])
        sound.play()
    except Exception as e:
        print(f"Audio error: {e}")

# Flatten landmarks for model input
def flatten_landmarks(landmarks):
    return [val for lm in landmarks for val in (lm.x, lm.y, lm.z)]

# Predict hand pose
def predict_hand_pose(landmarks):
    return model.predict([flatten_landmarks(landmarks)])[0]

# Get hand label
def get_hand_label(results):
    if results.multi_handedness:
        return results.multi_handedness[0].classification[0].label
    return None

# Draw progress bar
def draw_progress_bar(image, progress, y_offset=420, color=(255, 255, 0)):
    start_x, end_x = 50, 590
    filled = int(progress * (end_x - start_x))
    cv2.rectangle(image, (start_x, y_offset), (end_x, y_offset + 20), (50, 50, 50), 2)
    cv2.rectangle(image, (start_x, y_offset), (start_x + filled, y_offset + 20), color, -1)
    return image

# Draw text
def display_text_on_frame(frame, text, y, color=(255, 255, 255), scale=0.7):
    return cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# Hand session
def run_hand_session(cap, camera_feed, ax, hand_to_detect):
    flexion_start_time = extension_start_time = None
    flexion_done = extension_done = False
    result_time = None
    stage = f"Hold your {hand_to_detect.lower()} hand to the screen..."
    countdown_start = time.time()

    audio_played = {
        "hand_detected": False,
        "fist_held": False,
        "extension_done": False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        elapsed = time.time() - countdown_start
        time_left = max(0, 60 - int(elapsed))
        progress = 0

        if results.multi_hand_landmarks:
            label = get_hand_label(results)
            if label == hand_to_detect:
                # Play detection audio once
                if not audio_played["hand_detected"]:
                    play_audio("left" if hand_to_detect == "Left" else "right")
                    audio_played["hand_detected"] = True

                landmarks = results.multi_hand_landmarks[0].landmark
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                pose = predict_hand_pose(landmarks)

                if not flexion_done and pose == FLEXION_CLASS:
                    if flexion_start_time is None:
                        flexion_start_time = time.time()
                    progress = min((time.time() - flexion_start_time) / 1.0, 1.0)
                    if progress >= 1.0:
                        flexion_done = True
                        stage = "Fist held steady. Now extend!"
                        if not audio_played["fist_held"]:
                            play_audio("fist")
                            audio_played["fist_held"] = True

                elif flexion_done and not extension_done and pose == EXTENSION_CLASS:
                    if extension_start_time is None:
                        extension_start_time = time.time()
                    progress = min((time.time() - extension_start_time) / 1.0, 1.0)
                    if progress >= 1.0:
                        extension_done = True
                        result_time = round(time.time() - flexion_start_time, 2)
                        stage = f"{hand_to_detect} hand done in {result_time} seconds"
                        if not audio_played["extension_done"]:
                            play_audio("extension")
                            audio_played["extension_done"] = True
                        break
                else:
                    # Reset timers if pose lost
                    if not flexion_done:
                        flexion_start_time = None
                    elif not extension_done:
                        extension_start_time = None
            else:
                stage = f"Show your {hand_to_detect} hand"
                flexion_start_time = extension_start_time = None
        else:
            stage = f"Show your {hand_to_detect} hand"
            flexion_start_time = extension_start_time = None

        if elapsed > 60:
            stage = "DNF - Time exceeded"
            result_time = "DNF"
            break

        frame = display_text_on_frame(frame, f"Time Left: {time_left}s", 30, (0, 255, 0), 1)
        frame = display_text_on_frame(frame, stage, 460)
        frame = draw_progress_bar(frame, progress)

        camera_feed.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Finger Flexion and Extension", fontsize=16)
        for txt in ax.texts:
            txt.remove()
        ax.text(0.5, -0.15, stage, fontsize=12, ha='center', transform=ax.transAxes)
        plt.pause(0.001)

    return result_time

# Loading screen
def show_loading_screen(ax, camera_feed, cap):
    for _ in range(60):
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera feed not available during loading screen.")
            break
        frame = cv2.flip(frame, 1)
        frame[:] = 0
        cv2.putText(frame, "Loading...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        camera_feed.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Finger Flexion and Extension", fontsize=16)
        for txt in ax.texts:
            txt.remove()
        plt.pause(0.01)

# Main function
def main():
    # Camera initialization with error handling
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access camera")
        return

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('Finger Flexion and Extension')
    ax.axis('off')

    # Initial frame capture
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera.")
        cap.release()
        return

    frame = cv2.flip(frame, 1)
    camera_feed = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    show_loading_screen(ax, camera_feed, cap)

    left_result = run_hand_session(cap, camera_feed, ax, "Left")
    time.sleep(2)
    right_result = run_hand_session(cap, camera_feed, ax, "Right")
    time.sleep(2)

    # Final result screen
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame[:] = 0
        display_text_on_frame(frame, "Results", 100, (255, 255, 255), 1.5)
        display_text_on_frame(frame, f"Left Hand: {left_result}", 200, (100, 255, 100))
        display_text_on_frame(frame, f"Right Hand: {right_result}", 250, (100, 255, 100))

        camera_feed.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Finger Flexion and Extension - Results", fontsize=16)
        for txt in ax.texts:
            txt.remove()
        plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()