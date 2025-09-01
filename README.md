# Real-Time Hand Gesture Recognition

This project is a Python application that uses a webcam to recognise a predefined set of static hand gestures in real-time using computer vision and machine learning.

**Full Name:** Aditya  Sai Pranav Padakanti

## Demonstration

Below is a short demonstration of the application recognising various hand gestures.

[demo](https://drive.google.com/file/d/1P7x4wETYEyf2u02k8u1G65Pu_ZcJmH9T/view?usp=drive_link)

## Technology Justification

-   **OpenCV**: Used for capturing the live video feed from the webcam, handling image processing tasks, and displaying the video stream with annotations (landmarks, bounding boxes, and text). It is the industry standard for computer vision tasks in Python.
-   **MediaPipe**: Chosen for its high-fidelity hand and finger tracking solution. It provides 21 3D hand landmarks in real-time with minimal latency and high accuracy, which is crucial for feature extraction. This pre-trained model saves significant effort compared to building a hand detector from scratch.
-   **Scikit-learn**: Used for implementing the machine learning pipeline. This project uses a K-Nearest Neighbours (KNN) classifier to predict gestures based on the landmark data. Scikit-learn offers a straightforward and efficient implementation of various classifiers, facilitating the training of models and their subsequent performance evaluation.

## Gesture Logic Explanation

This project has transitioned from a simple rule-based system to a more robust machine learning approach.

1.  **Data Preparation (`src/prepare_data.py`)**:
    -   The process starts with a collection of gesture images located in `data/raw`, categorised into subfolders by gesture name (e.g., `fist`, `open_palm`).
    -   The `prepare_data.py` script iterates through these images, using MediaPipe to detect hand landmarks.
    -   For each detected hand, the 21 landmarks are **normalised**. Normalisation makes the model robust to variations in hand size and position relative to the camera. The script calculates the coordinates relative to the wrist and scales them.
    -   The normalised landmark data and its corresponding label are saved into a single CSV file: `data/processed/landmarks.csv`.
    

2.  **Model Training (`src/train_knn.py`)**:
    -   The `train_knn.py` script reads the `landmarks.csv` file.
    -   It splits the data into a training set and a testing set.
    -   A K-Nearest Neighbours (KNN) classifier is trained on the landmark data. The features are also scaled using `StandardScaler` to ensure all landmarks contribute equally to the distance calculations.
    -   The trained KNN model and the scaler are saved to the `models/` directory as `knn_model.pkl` and `knn_scaler.pkl`.

3.  **Real-Time Prediction (`src/main.py`)**:
    -   The main application loads the pre-trained KNN model and scaler.
    -   It captures video from the webcam frame by frame.
    -   For each frame, MediaPipe detects hands and extracts their landmarks.
    -   The landmarks are normalised and scaled in the same way as the training data.
    -   The processed data is fed into the KNN model, which predicts the gesture.
    -   The recognised gesture name is then displayed on the screen over the detected hand.

## Gesture Vocabulary

The model is trained to recognise the following gestures based on the folders in `data/raw`:
-   Fist
-   Open Palm
-   Peace Sign (Victory)
-   Thumbs Up
-   Call

## Setup and Execution Instructions

### 1. Prerequisites
- Python 3.8+
- A webcam

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd hand_gesture_recognizer
```

### 3. Set Up a Virtual Environment (Recommended)
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Pipeline
Follow these steps to process data, train the model, and run the application.

**Step 1: Prepare the Landmark Data**
This script will process the images in `data/raw` and create `data/processed/landmarks.csv`.
```bash
python3 src/prepare_data.py
```

**Step 2: Train the KNN Model**
This script trains the model on the landmark data and saves it to the `models/` directory.
```bash
python3 src/train_knn.py
```

**Step 3: Run the Gesture Recognition App**
This will start the application using the trained KNN model.
```bash
python3 src/main.py 
```
Press the 'q' key to quit the application.
