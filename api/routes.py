"""
API route definitions for the meal recommendation system
"""

from flask import request, jsonify, send_file, current_app, make_response
import os
import pickle
import pandas as pd
import logging
import traceback
from models.meal_recommendation import ImprovedMealRecommendationModel
from utils.serializers import convert_to_serializable

# Configure logger
logger = logging.getLogger(__name__)

def register_routes(app, global_model):
    """
    Register all API routes
    
    Args:
        app: Flask application instance
        global_model: Global model variable from app.py
    """
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """
        Simple health check endpoint to verify the API is running
        """
        return jsonify({
            "status": "healthy",
            "message": "Meal Recommendation API is running",
            "model_loaded": global_model is not None
        })

    @app.route('/api/train', methods=['POST'])
    def train_model():
        """
        Train the model with a CSV file
        
        Request: multipart/form-data with a CSV file
        Response: JSON with training results
        """
        nonlocal global_model
        
        try:
            # Check if file is in the request
            if 'file' not in request.files:
                return jsonify({
                    "error": "No file provided",
                    "status": "error"
                }), 400
            
            file = request.files['file']
            
            # Check if filename is empty
            if file.filename == '':
                return jsonify({
                    "error": "No file selected",
                    "status": "error"
                }), 400
            
            # Check if the file is a CSV
            if not file.filename.endswith('.csv'):
                return jsonify({
                    "error": "File must be a CSV",
                    "status": "error"
                }), 400
            
            # Save the file temporarily
            temp_path = 'temp_training_data.csv'
            file.save(temp_path)
            
            try:
                # Load the data
                logger.info(f"Loading data from {temp_path}")
                data = pd.read_csv(temp_path)
                
                # Create and train the model
                logger.info("Creating and training the model")
                model = ImprovedMealRecommendationModel(k=15)
                model.train(data)
                
                # Update the global model
                global_model = model
                
                # Save the model to disk
                logger.info("Saving model to disk")
                with open('trained_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                
                return jsonify({
                    "status": "success",
                    "message": f"Model trained successfully on {len(data)} rows",
                    "accuracy": float(model.model_accuracy) if model.model_accuracy is not None else None
                })
            finally:
                # Remove the temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/api/predict', methods=['POST', 'OPTIONS'])
    def predict():
        """
        Get meal recommendations for user data
        
        Request: JSON with user data
        Response: JSON with meal recommendations
        """
        nonlocal global_model

        # Handle preflight request
        if request.method == 'OPTIONS':
            response = make_response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response
        
        try:
            logger.info("Received prediction request")
            logger.debug(f"Request data: {request.json}")
            
            # Check if model is trained
            if global_model is None:
                logger.warning("Model not loaded, attempting to load from disk")
                # Try to load from disk if available
                if os.path.exists('trained_model.pkl'):
                    try:
                        with open('trained_model.pkl', 'rb') as f:
                            global_model = pickle.load(f)
                        logger.info("Successfully loaded model from disk")
                    except Exception as e:
                        logger.error(f"Failed to load model: {str(e)}")
                        error_response = jsonify({
                            "error": "Could not load model from disk",
                            "status": "error"
                        })
                        error_response.headers.add('Access-Control-Allow-Origin', '*')
                        return error_response, 500
                else:
                    logger.error("No trained model available")
                    error_response = jsonify({
                        "error": "Model not trained. Please train the model first.",
                        "status": "error"
                    })
                    error_response.headers.add('Access-Control-Allow-Origin', '*')
                    return error_response, 400
            
            # Get user data from request
            user_data = request.json
            
            # Validate required fields
            required_fields = [
                "age", "gender", "height", "weight", 
                "activity_level", "dietary_preference"
            ]
            
            for field in required_fields:
                if field not in user_data:
                    logger.warning(f"Missing required field: {field}")
                    error_response = jsonify({
                        "error": f"Missing required field: {field}",
                        "status": "error"
                    })
                    error_response.headers.add('Access-Control-Allow-Origin', '*')
                    return error_response, 400
            
            # Set default values for optional fields
            if "has_diabetes" not in user_data:
                user_data["has_diabetes"] = False
            if "has_hypertension" not in user_data:
                user_data["has_hypertension"] = False
            if "has_heart_disease" not in user_data:
                user_data["has_heart_disease"] = False
            if "has_kidney_disease" not in user_data:
                user_data["has_kidney_disease"] = False
            if "has_weight_gain" not in user_data:
                user_data["has_weight_gain"] = False
            if "has_weight_loss" not in user_data:
                user_data["has_weight_loss"] = False
            if "has_acne" not in user_data:
                user_data["has_acne"] = False
            
            # Ensure numeric types
            try:
                user_data["age"] = int(user_data["age"])
                user_data["height"] = float(user_data["height"])
                user_data["weight"] = float(user_data["weight"])
                user_data["activity_level"] = int(user_data["activity_level"])
                user_data["dietary_preference"] = int(user_data["dietary_preference"])
            except ValueError as e:
                logger.error(f"Type conversion error: {str(e)}")
                error_response = jsonify({
                    "error": f"Invalid data type: {str(e)}",
                    "status": "error"
                })
                error_response.headers.add('Access-Control-Allow-Origin', '*')
                return error_response, 400
                
            # Encode user input if needed
            try:
                logger.debug("Encoding user input")
                encoded_user_input = global_model.encode_user_input(user_data)
            except Exception as e:
                logger.error(f"Error encoding user input: {str(e)}")
                logger.error(traceback.format_exc())
                error_response = jsonify({
                    "error": f"Error encoding user data: {str(e)}",
                    "status": "error"
                })
                error_response.headers.add('Access-Control-Allow-Origin', '*')
                return error_response, 500
            
            # Get recommendations
            try:
                logger.debug("Getting recommendations")
                recommendations = global_model.predict(encoded_user_input)
                logger.debug("Successfully generated recommendations")
            except Exception as e:
                logger.error(f"Error predicting recommendations: {str(e)}")
                logger.error(traceback.format_exc())
                error_response = jsonify({
                    "error": f"Error generating recommendations: {str(e)}",
                    "status": "error"
                })
                error_response.headers.add('Access-Control-Allow-Origin', '*')
                return error_response, 500
            
            # Convert to serializable format
            try:
                logger.debug("Converting recommendations to serializable format")
                serializable_recommendations = convert_to_serializable(recommendations)
            except Exception as e:
                logger.error(f"Error serializing recommendations: {str(e)}")
                logger.error(traceback.format_exc())
                error_response = jsonify({
                    "error": f"Error processing results: {str(e)}",
                    "status": "error"
                })
                error_response.headers.add('Access-Control-Allow-Origin', '*')
                return error_response, 500
            
            logger.info("Successfully processed prediction request")
            response = jsonify({
                "status": "success",
                "recommendations": serializable_recommendations
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error in predict: {str(e)}")
            logger.error(traceback.format_exc())
            error_response = jsonify({
                "error": str(e),
                "status": "error"
            })
            error_response.headers.add('Access-Control-Allow-Origin', '*')
            return error_response, 500

    @app.route('/api/download_model', methods=['GET'])
    def download_model():
        """
        Download the trained model
        """
        if not os.path.exists('trained_model.pkl'):
            return jsonify({
                "error": "No trained model available",
                "status": "error"
            }), 404
        
        return send_file('trained_model.pkl', 
                        mimetype='application/octet-stream',
                        as_attachment=True,
                        download_name='meal_recommendation_model.pkl')