import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

all_data = []
labels = []

print("Press 'f' to save fist, 'o' to save open hand, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            flat_landmarks = []
            for lm in hand_landmarks.landmark:
                flat_landmarks.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(10) & 0xFF
            if key == ord('f'):  # Fist
                all_data.append(flat_landmarks)
                labels.append(0)
                print("Fist recorded")
            elif key == ord('o'):  # Open hand
                all_data.append(flat_landmarks)
                labels.append(1)
                print("Open hand recorded")
    else:
        key = cv2.waitKey(10) & 0xFF

    cv2.imshow("Data Collection", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(all_data)
df['label'] = labels
df.to_csv("flexion_extension_data.csv", index=False)
print("Dataset saved to flexion_extension_data.csv")