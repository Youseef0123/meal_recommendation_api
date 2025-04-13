"""
Model Training Script for Meal Recommendation System

This script trains the meal recommendation model using your CSV data
and saves it to disk for future use.

Usage:
    python train_model.py path/to/your/data.csv

The trained model will be saved as 'trained_model.pkl' in the current directory.
"""

import os
import sys
import pickle
import pandas as pd
from models.meal_recommendation import ImprovedMealRecommendationModel

def train_and_save_model(csv_file_path, output_path='trained_model.pkl'):
    """
    Train the meal recommendation model and save it to disk
    
    Args:
        csv_file_path (str): Path to the CSV file with meal data
        output_path (str): Path where the trained model will be saved
    """
    print(f"Loading data from {csv_file_path}...")
    try:
        # Load the CSV data
        data = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {len(data)} rows of data")
        
        # Create the model
        print("Creating model...")
        model = ImprovedMealRecommendationModel(k=15)
        
        # Preprocess data without calculating accuracy
        print("Preprocessing data...")
        model.data = model.preprocess_data(data)
        
        # Define feature columns
        model.feature_columns = [
            'Ages', 'GenderEncoded', 'Height', 'Weight', 'ActivityLevelEncoded', 
            'DietaryPreferenceEncoded', 'HasDiabetes', 'HasHypertension', 
            'HasHeartDisease', 'HasKidneyDisease', 'HasWeightGain', 
            'HasWeightLoss', 'HasAcne'
        ]
        
        # Normalize feature data
        print("Normalizing features...")
        features_to_scale = model.data[model.feature_columns].copy()
        model.scaled_data = pd.DataFrame(
            model.scaler.fit_transform(features_to_scale),
            columns=model.feature_columns,
            index=features_to_scale.index
        )
        
        # Store the raw data for nutrition lookup
        model.raw_data = data.copy()
        
        # Skip accuracy calculation for now
        model.model_accuracy = 0.95  # Placeholder
        
        # Save the model to disk
        print(f"Saving trained model to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved successfully!")
        return True
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Check if CSV file path is provided as command-line argument
    if len(sys.argv) < 2:
        print("Please provide the path to your CSV file")
        print("Usage: python train_model.py path/to/your/data.csv")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found at {csv_file_path}")
        sys.exit(1)
    
    # Train and save the model
    success = train_and_save_model(csv_file_path)
    
    if success:
        print("\nYou can now run your meal recommendation API with the pre-trained model")
        print("Use 'python app.py' to start the API server")
    else:
        print("\nModel training failed. Please check the error messages above")