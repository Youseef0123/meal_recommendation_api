"""
Google Cloud Storage utilities for model storage
"""
import pickle
import tempfile
from google.cloud import storage

def save_model_to_gcs(model, bucket_name, blob_name='trained_model.pkl'):
    """
    Save model to Google Cloud Storage
    
    Args:
        model: The trained model to save
        bucket_name (str): GCS bucket name
        blob_name (str): Name of the blob (file) in the bucket
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Save model to a temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl') as temp_file:
        pickle.dump(model, temp_file)
        temp_file.flush()
        # Upload to GCS
        blob.upload_from_filename(temp_file.name)

def load_model_from_gcs(bucket_name, blob_name='trained_model.pkl'):
    """
    Load model from Google Cloud Storage
    
    Args:
        bucket_name (str): GCS bucket name
        blob_name (str): Name of the blob (file) in the bucket
        
    Returns:
        The loaded model
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download to a temporary file
    with tempfile.NamedTemporaryFile(mode='rb+', suffix='.pkl') as temp_file:
        blob.download_to_filename(temp_file.name)
        temp_file.seek(0)
        # Load the model
        return pickle.load(temp_file)