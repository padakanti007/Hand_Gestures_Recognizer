# import cv2
# import mediapipe as mp # type: ignore
# import cv2

# def draw_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#     """
#     Draws text on an image.
#     """
#     cv2.putText(img, text, position, font, font_scale, color, thickness)

# class GestureRecognizer:
#     def __init__(self, landmark_list):
#         self.lm_list = landmark_list
#         self.tip_ids = [4, 8, 12, 16, 20]

#     def recognize(self):
#         if not self.lm_list:
#             return "No Hand"

#         fingers = self._get_finger_status()
        
#         # Open Palm / All fingers up
#         if all(fingers):
#             return "Open Palm"
#         # Fist / All fingers down
#         elif not any(fingers):
#             return "Fist"
#         # Victory (Peace Sign)
#         elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:#if index and middle fingers are up
#             return "Victory"
#         # Thumbs Up
#         elif fingers[0] and not any(fingers[1:]):
#             return "Thumbs Up"
#         # Thumbs Down
#         elif not fingers[0] and not any(fingers[1:]) and self._is_thumb_down():
#              return "Thumbs Down"
#         # Pointing Up
#         elif fingers[1] and not any([fingers[0]] + fingers[2:]):
#             return "Pointing Up"
#         # Love
#         elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
#             return "Love"
            
#         return "Unknown"

#     def _is_thumb_down(self):
#         # Check if thumb tip is below the MCP joint of the thumb
#         return self.lm_list[self.tip_ids[0]][2] > self.lm_list[self.tip_ids[0] - 2][2]

#     def _get_finger_status(self):
#         fingers = []
        
#         # Thumb
#         if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
#             fingers.append(1)
#         else:
#             fingers.append(0)

#         # 4 Fingers
#         for id in range(1, 5):
#             if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
#                 fingers.append(1)
#             else:
#                 fingers.append(0)
        
#         return fingers

# class HandTracker:
#     def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
#         self.mode = mode
#         self.max_hands = max_hands
#         self.detection_con = detection_con
#         self.track_con = track_con

#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=self.mode,
#             max_num_hands=self.max_hands,
#             min_detection_confidence=self.detection_con,
#             min_tracking_confidence=self.track_con
#         )
#         self.mp_draw = mp.solutions.drawing_utils
#         self.tip_ids = [4, 8, 12, 16, 20]

#     def find_hands(self, img, draw=True):
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(img_rgb)

#         if self.results.multi_hand_landmarks:
#             for hand_lms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
#         return img

#     def find_position(self, img, hand_no=0, draw=True):
#         lm_list = []
#         if self.results.multi_hand_landmarks:
#             my_hand = self.results.multi_hand_landmarks[hand_no]
#             for id, lm in enumerate(my_hand.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lm_list.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#         return lm_list

# def main():
#     cap = cv2.VideoCapture(0)
#     tracker = HandTracker(max_hands=2)

#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         # Find hands and draw landmarks
#         img = tracker.find_hands(img, draw=True)

#         if tracker.results.multi_hand_landmarks:
#             # Iterate over each detected hand
#             for hand_landmarks in tracker.results.multi_hand_landmarks:
#                 # Get landmark positions for the current hand
#                 lm_list = []
#                 x_coords = []
#                 y_coords = []
#                 h, w, c = img.shape

#                 for id, lm in enumerate(hand_landmarks.landmark):
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     lm_list.append([id, cx, cy])
#                     x_coords.append(cx)
#                     y_coords.append(cy)

#                 if lm_list:
#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
                    
#                     padding = 20
#                     # Draw bounding box
#                     cv2.rectangle(img, (x_min - padding, y_min - padding), (x_max + padding, y_max + padding), (0, 255, 0), 2)

#                     # Recognize gesture
#                     recognizer = GestureRecognizer(lm_list)
#                     gesture = recognizer.recognize()
                    
#                     # Display gesture name
#                     draw_text(img, gesture, (x_min - padding, y_min - padding - 10), font_scale=1, color=(0, 255, 0), thickness=2)

#         cv2.imshow("Image", img)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import cv2
import mediapipe as mp
import numpy as np
import joblib

# --- HandTracker Class (Modified to default to 2 hands) ---
class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5): # Default max_hands=2
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

# --- Feature Extraction Function (Normalized) ---
def extract_features(hand_landmarks):
    """
    Normalizes hand landmarks to be invariant to position and scale.
    """
    origin = hand_landmarks.landmark[0]
    relative_landmarks = []
    for lm in hand_landmarks.landmark:
        relative_landmarks.extend([lm.x - origin.x, lm.y - origin.y, lm.z - origin.z])
    
    max_val = max(map(abs, relative_landmarks))
    if max_val == 0:
        return np.zeros(len(relative_landmarks))
        
    normalized_landmarks = np.array(relative_landmarks) / max_val
    return normalized_landmarks.flatten()

# --- Main Application Logic ---
def main():
    # 1. Load the trained model and scaler
    MODEL_PATH = 'models/knn_model.pkl'
    SCALER_PATH = 'models/scaler.pkl' # Make sure you have a saved scaler from training
    try:
        knn_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH) # Load the scaler
        print(f"Model and scaler loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model or scaler file not found.")
        print("Please run train_knn.py to train and save them first.")
        return

    # Initialize webcam and HandTracker for two hands
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=2) # ✨ MODIFIED: Detect up to 2 hands

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)

        # Find hands and draw landmarks
        img = tracker.find_hands(img, draw=True)

        # ✨ MODIFIED: Loop through each detected hand
        if tracker.results.multi_hand_landmarks:
            for hand_landmarks in tracker.results.multi_hand_landmarks:
                
                # --- Bounding Box Calculation ---
                h, w, c = img.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                padding = 25
                # Draw bounding box
                cv2.rectangle(img, (x_min - padding, y_min - padding), 
                              (x_max + padding, y_max + padding), (0, 255, 0), 2)

                # --- Gesture Prediction for the current hand ---
                features = extract_features(hand_landmarks)
                features_reshaped = features.reshape(1, -1)
                
                # Scale the features before prediction
                features_scaled = scaler.transform(features_reshaped)
                
                gesture = "Unknown"
                try:
                    prediction = knn_model.predict(features_scaled)
                    gesture = prediction[0]
                except Exception as e:
                    print(f"Error during prediction: {e}")

                # --- Display the result above the bounding box ---
                cv2.putText(img, gesture, (x_min - padding, y_min - padding - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition (k-NN)", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()