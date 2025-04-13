"""
Google Cloud Storage utilities for model storage
"""
import pickle
import tempfile
import logging
from google.cloud import storage
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)

def save_model_to_gcs(model, bucket_name, blob_name='trained_model.pkl'):
    """
    Save model to Google Cloud Storage
    
    Args:
        model: The trained model to save
        bucket_name (str): GCS bucket name
        blob_name (str): Name of the blob (file) in the bucket
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl') as temp_file:
            pickle.dump(model, temp_file)
            temp_file.flush()
            # Upload to GCS
            blob.upload_from_filename(temp_file.name)
        logger.info(f"Successfully saved model to GCS bucket: {bucket_name}/{blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error saving model to GCS: {str(e)}")
        return False

def load_model_from_gcs(bucket_name, blob_name='trained_model.pkl'):
    """
    Load model from Google Cloud Storage
    
    Args:
        bucket_name (str): GCS bucket name
        blob_name (str): Name of the blob (file) in the bucket
        
    Returns:
        The loaded model or None if not found
    """
    try:
        logger.info(f"Attempting to load model from {bucket_name}/{blob_name}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists
        try:
            bucket.reload()
        except NotFound:
            logger.error(f"Bucket not found: {bucket_name}")
            return None
        
        # Check if blob exists
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.error(f"Blob not found: {blob_name}")
            return None
        
        # Download and load the model
        with tempfile.NamedTemporaryFile(mode='wb+', suffix='.pkl') as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_file.seek(0)
            model = pickle.load(temp_file)
            logger.info(f"Successfully loaded model from GCS")
            return model
    except Exception as e:
        logger.error(f"Error loading model from GCS: {str(e)}")
        return None