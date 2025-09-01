import os
import cv2
import mediapipe as mp
import numpy as np
import csv

def process_data():
    # Setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    RAW_DATA_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, 'landmarks.csv')
    
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Get gesture categories from folder names
    gestures = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    with open(PROCESSED_CSV_PATH, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Create header row for CSV
        header = ['label']
        for i in range(21):
            header += [f'x{i}', f'y{i}', f'z{i}']
        csv_writer.writerow(header)
        
        # Process each image
        for gesture in gestures:
            raw_gesture_path = os.path.join(RAW_DATA_DIR, gesture)
            processed_gesture_path = os.path.join(PROCESSED_DIR, gesture)
            os.makedirs(processed_gesture_path, exist_ok=True)

            for filename in os.listdir(raw_gesture_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(raw_gesture_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # 1. Save annotated image
                            annotated_img = img.copy()
                            mp_drawing.draw_landmarks(annotated_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            output_path = os.path.join(processed_gesture_path, filename)
                            cv2.imwrite(output_path, annotated_img)

                            # 2. Save landmarks to CSV
                            first_landmark = hand_landmarks.landmark[0]
                            base_x, base_y, base_z = first_landmark.x, first_landmark.y, first_landmark.z
                            
                            row = [gesture]
                            for lm in hand_landmarks.landmark:
                                row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                            
                            csv_writer.writerow(row)
                            
    print(f"Data processing complete. Landmarks saved to {PROCESSED_CSV_PATH}")
    print(f"Annotated images saved in {PROCESSED_DIR}")

if __name__ == '__main__':
    process_data()
