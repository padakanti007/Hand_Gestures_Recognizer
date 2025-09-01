import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # 1. Import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    # Load data
    data = pd.read_csv('data/processed/landmarks.csv')
    
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN model on the SCALED data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate model on the SCALED data
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # --- Save BOTH the model and the scaler ---
    MODEL_PATH = 'models/knn_model.pkl'
    SCALER_PATH = 'models/scaler.pkl' # 3. Define path for the scaler
    
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH) # 4. Save the scaler object
    
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}") # You will now see this confirmation message

if __name__ == '__main__':
    train_model()