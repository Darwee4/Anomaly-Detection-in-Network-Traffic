import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class NetworkAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        
    def load_data(self, dataset_path):
        """Load and preprocess the UNSW-NB15 dataset"""
        # Load dataset
        data = pd.read_csv(dataset_path)
        
        # Feature selection - using key network traffic features
        features = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 
            'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 
            'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 
            'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 
            'synack', 'ackdat', 'smean', 'dmean'
        ]
        
        # Preprocessing
        X = data[features]
        X = pd.get_dummies(X, columns=['proto', 'service', 'state'])
        X = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test = train_test_split(X, test_size=0.2, random_state=42)
        
    def build_model(self):
        """Build autoencoder model"""
        input_dim = self.X_train.shape[1]
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(64, activation="relu")(input_layer)
        encoder = Dense(32, activation="relu")(encoder)
        encoder = Dense(16, activation="relu")(encoder)
        
        # Decoder
        decoder = Dense(32, activation="relu")(encoder)
        decoder = Dense(64, activation="relu")(decoder)
        decoder = Dense(input_dim, activation=None)(decoder)
        
        # Autoencoder
        self.model = Model(inputs=input_layer, outputs=decoder)
        self.model.compile(optimizer='adam', loss='mse')
        
    def train(self, epochs=50, batch_size=128):
        """Train the autoencoder"""
        history = self.model.fit(
            self.X_train, self.X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.X_test),
            shuffle=True
        )
        
        # Set anomaly detection threshold
        reconstructions = self.model.predict(self.X_train)
        train_loss = tf.keras.losses.mse(reconstructions, self.X_train)
        self.threshold = np.mean(train_loss) + 2 * np.std(train_loss)
        
    def detect_anomalies(self, real_time_data):
        """Detect anomalies in real-time network traffic"""
        if self.model is None or self.threshold is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Preprocess input data
        processed_data = self.scaler.transform(real_time_data)
        
        # Get reconstruction error
        reconstructions = self.model.predict(processed_data)
        loss = tf.keras.losses.mse(reconstructions, processed_data)
        
        # Return anomaly predictions
        return loss > self.threshold

if __name__ == "__main__":
    detector = NetworkAnomalyDetector()
    
    # Load and preprocess data
    detector.load_data("UNSW-NB15.csv")
    
    # Build and train model
    detector.build_model()
    detector.train()
    
    # Example usage for real-time detection
    # real_time_data = [...]  # New network traffic data
    # anomalies = detector.detect_anomalies(real_time_data)
    # print("Anomalies detected:", anomalies)
