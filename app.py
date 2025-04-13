"""
Main application entry point
Runs the Flask server and registers API routes
"""

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import os
import pickle
import pandas as pd
import numpy as np
import logging
import datetime
import traceback
import datetime
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the application
app = Flask(__name__)

# Configure CORS properly - this should be done only once
CORS(app, resources={r"/api/*": {"origins": "*"}}, 
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Make sure all API responses include CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Global variable to store the trained model
trained_model = None

# استيراد وظائف التخزين السحابي
from utils.gcs_storage import load_model_from_gcs

# اسم الخزانة (Bucket)
GCS_BUCKET_NAME = 'global-sun-456710-t3-models'
# تحميل النموذج المدرب
trained_model = None
model_path = 'trained_model.pkl'

# First try to load from GCS
try:
    logger.info(f"Attempting to load model from GCS bucket: {GCS_BUCKET_NAME}")
    trained_model = load_model_from_gcs(GCS_BUCKET_NAME)
    if trained_model is not None:
        logger.info(f"Successfully loaded trained model from GCS bucket")
    else:
        logger.warning(f"No model found in GCS bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    logger.warning(f"Error loading from GCS: {str(e)}")

# If GCS loading failed, try local file
if trained_model is None and os.path.exists(model_path):
    try:
        logger.info(f"Attempting to load model from local file: {model_path}")
        with open(model_path, 'rb') as f:
            trained_model = pickle.load(f)
        logger.info(f"Successfully loaded trained model from local file")
    except Exception as e:
        logger.error(f"Error loading model from local file: {str(e)}")

# If still no model, create a dummy model for testing
if trained_model is None:
    logger.warning("No model found. Creating a dummy model for testing")
    from models.meal_recommendation import ImprovedMealRecommendationModel
    trained_model = ImprovedMealRecommendationModel(k=15)
    
    # Add dummy data structure# If still no model, create a dummy model for testing
if trained_model is None:
    logger.warning("No model found. Creating a dummy model for testing")
    from models.meal_recommendation import ImprovedMealRecommendationModel
    from sklearn.preprocessing import StandardScaler
    
    # Create the model
    trained_model = ImprovedMealRecommendationModel(k=15)
    
    # Initialize the scaler with some dummy data
    dummy_data = np.array([
        [30, 1, 170, 70, 2, 0, 0, 0, 0, 0, 0, 0, 0],  # Example user 1
        [25, 0, 160, 60, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Example user 2
        [40, 1, 180, 80, 3, 0, 0, 1, 0, 0, 0, 0, 0],  # Example user 3
        [35, 0, 165, 65, 2, 2, 0, 0, 1, 0, 0, 0, 0],  # Example user 4
        [50, 1, 175, 75, 1, 3, 0, 0, 0, 1, 0, 0, 0]   # Example user 5
    ])
    
    # Create feature columns
    trained_model.feature_columns = [
        'Ages', 'GenderEncoded', 'Height', 'Weight', 'ActivityLevelEncoded', 
        'DietaryPreferenceEncoded', 'HasDiabetes', 'HasHypertension', 
        'HasHeartDisease', 'HasKidneyDisease', 'HasWeightGain', 
        'HasWeightLoss', 'HasAcne'
    ]
    
    # Fit the scaler with dummy data
    trained_model.scaler = StandardScaler()
    trained_model.scaler.fit(dummy_data)
    
    # Create scaled data
    trained_model.scaled_data = pd.DataFrame(
        trained_model.scaler.transform(dummy_data),
        columns=trained_model.feature_columns
    )
    
    # Add dummy raw_data with meal suggestions
    columns = ['Ages', 'Gender', 'Height', 'Weight', 'Activity Level', 'Dietary Preference',
               'Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion',
               'Breakfast Calories', 'Breakfast Protein', 'Breakfast Carbohydrates', 'Breakfast Fats',
               'Lunch Calories', 'Lunch Protein', 'Lunch Carbohydrates', 'Lunch Fats',
               'Dinner Calories', 'Dinner Protein.1', 'Dinner Carbohydrates.1', 'Dinner Fats',
               'Snacks Calories', 'Snacks Protein', 'Snacks Carbohydrates', 'Snacks Fats',
               'Disease']
    
    # Create dummy data with a few meal options
    data = []
    for i in range(5):
        row = [30, 'Male', 170, 70, 'Moderately Active', 'Omnivore', 
               'oatmeal with berries', 'chicken salad', 'grilled salmon with vegetables', 'yogurt with fruits',
               300, 10, 45, 5,  # Breakfast nutrition
               400, 30, 30, 15,  # Lunch nutrition
               500, 35, 40, 20,  # Dinner nutrition
               150, 8, 20, 5,    # Snack nutrition
               '']
        data.append(row)
    
    # Add some vegetarian options
    for i in range(3):
        row = [30, 'Female', 160, 60, 'Lightly Active', 'Vegetarian',
               'avocado toast', 'quinoa salad', 'vegetable stir-fry', 'hummus with vegetables',
               250, 8, 30, 12,   # Breakfast nutrition
               350, 15, 45, 10,  # Lunch nutrition 
               400, 18, 50, 15,  # Dinner nutrition
               120, 5, 15, 8,    # Snack nutrition
               '']
        data.append(row)
    
    # Create DataFrame
    trained_model.raw_data = pd.DataFrame(data, columns=columns)
    
    # Create processed data
    trained_model.data = trained_model.preprocess_data(trained_model.raw_data)
    
    # Set activity multipliers
    trained_model.activity_multipliers = {
        0: 1.2,  # Sedentary
        1: 1.375,  # Lightly Active
        2: 1.55,  # Moderately Active
        3: 1.725,  # Very Active
        4: 1.9  # Extremely Active
    }
    
    # Set model accuracy
    trained_model.model_accuracy = 0.85
    
    logger.info("Created dummy model with initialized scaler as fallback")
    trained_model.raw_data = None
    trained_model.activity_multipliers = {
        0: 1.2,  # Sedentary
        1: 1.375,  # Lightly Active
        2: 1.55,  # Moderately Active
        3: 1.725,  # Very Active
        4: 1.9  # Extremely Active
    }
    logger.info("Created dummy model as fallback")

# Add the missing get_meal_nutrition method to the model
if trained_model is not None and not hasattr(trained_model, 'get_meal_nutrition'):
    logger.warning("Adding missing get_meal_nutrition method to the model")
    
    def get_meal_nutrition(self, meal_name, meal_type):
        """
        Get nutritional information for a meal
        
        Args:
            meal_name (str): Name of the meal
            meal_type (str): Type of meal (breakfast, lunch, dinner, snack)
            
        Returns:
            dict: Nutritional information
        """
        # Safety check for raw_data
        if self.raw_data is None:
            return {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
            
        # Define column mappings for each meal type
        nutrition_columns = {
            'breakfast': {
                'calories': 'Breakfast Calories',
                'protein': 'Breakfast Protein',
                'carbs': 'Breakfast Carbohydrates',
                'fats': 'Breakfast Fats'
            },
            'lunch': {
                'calories': 'Lunch Calories',
                'protein': 'Lunch Protein',
                'carbs': 'Lunch Carbohydrates',
                'fats': 'Lunch Fats'
            },
            'dinner': {
                'calories': 'Dinner Calories',
                'protein': 'Dinner Protein.1',
                'carbs': 'Dinner Carbohydrates.1',
                'fats': 'Dinner Fats'
            },
            'snack': {
                'calories': 'Snacks Calories',
                'protein': 'Snacks Protein',
                'carbs': 'Snacks Carbohydrates',
                'fats': 'Snacks Fats'
            }
        }
        
        # Safety check
        if meal_type.lower() not in nutrition_columns:
            return {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
        
        meal_col = f"{meal_type.capitalize()} Suggestion"
        if meal_type.lower() == 'snack':
            meal_col = "Snack Suggestion"
            
        # Normalize meal name for case-insensitive lookup
        meal_name = str(meal_name).lower().strip()
            
        # Find the meal in the dataset - using normalized meal name
        try:
            # Create a normalized column for comparison
            normalized_col = f"{meal_col}_normalized"
            temp_df = self.raw_data.copy()
            temp_df[normalized_col] = temp_df[meal_col].str.lower().str.strip()
            
            # Find matching meals
            meal_rows = temp_df[temp_df[normalized_col] == meal_name]
            
            if len(meal_rows) == 0:
                # Try fuzzy matching if exact match fails
                meal_rows = temp_df[temp_df[normalized_col].str.contains(meal_name, na=False)]
            
            if len(meal_rows) == 0:
                return {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
            
            # Get the first instance of this meal and extract nutrition info
            meal_row = meal_rows.iloc[0]
            cols = nutrition_columns[meal_type.lower()]
            
            # Handle missing values gracefully
            calories = meal_row[cols['calories']] if cols['calories'] in meal_row and pd.notna(meal_row[cols['calories']]) else 0
            protein = meal_row[cols['protein']] if cols['protein'] in meal_row and pd.notna(meal_row[cols['protein']]) else 0
            carbs = meal_row[cols['carbs']] if cols['carbs'] in meal_row and pd.notna(meal_row[cols['carbs']]) else 0
            fats = meal_row[cols['fats']] if cols['fats'] in meal_row and pd.notna(meal_row[cols['fats']]) else 0
            
            return {
                'calories': round(calories) if not np.isnan(calories) else 0,
                'protein': round(protein, 1) if not np.isnan(protein) else 0,
                'carbs': round(carbs, 1) if not np.isnan(carbs) else 0,
                'fats': round(fats, 1) if not np.isnan(fats) else 0
            }
        except Exception as e:
            logger.error(f"Error getting nutrition for {meal_name}: {e}")
            return {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
    
    # Add the method to the model
    import types
    trained_model.get_meal_nutrition = types.MethodType(get_meal_nutrition, trained_model)

# Helper function to convert NumPy types to Python types for JSON serialization
def convert_to_serializable(obj):
    """
    Convert object to JSON serializable format
    """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # For any other object, convert to string
        return str(obj)

# API health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify the API is running
    """
    return jsonify({
        "status": "healthy",
        "message": "Meal Recommendation API is running",
        "model_loaded": trained_model is not None
    })

@app.route('/api/system-check', methods=['GET'])
def system_check():
    """
    Enhanced health check with detailed system information
    """
    import platform
    import sys
    
    # Check if GCS credentials are accessible
    has_gcs_credentials = False
    try:
        from google.cloud import storage
        client = storage.Client()
        # Try listing buckets as a test
        list(client.list_buckets(max_results=1))
        has_gcs_credentials = True
    except Exception as e:
        logger.error(f"GCS credentials check failed: {str(e)}")
    
    # Check if model is loaded
    model_status = "loaded" if trained_model is not None else "not_loaded"
    
    # Check if local model file exists
    local_model_exists = os.path.exists('trained_model.pkl')
    
    # Get environment information
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_variables": {
            k: v for k, v in os.environ.items() 
            if k in ['GCS_BUCKET_NAME', 'GOOGLE_CLOUD_PROJECT', 'PORT']
        }
    }
    
    return jsonify({
        "status": "healthy",
        "message": "System check completed",
        "model_status": model_status,
        "local_model_exists": local_model_exists,
        "gcs_credentials_valid": has_gcs_credentials,
        "environment": env_info,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_route():
    """
    Simple test endpoint that doesn't rely on model
    """
    return jsonify({
        "status": "success",
        "message": "API is working",
        "timestamp": datetime.datetime.now().isoformat()
    })

# API prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Get meal recommendations for user data
    
    Request: JSON with user data
    Response: JSON with meal recommendations
    """
    global trained_model
    
    try:
        logger.info("Received prediction request")
        logger.debug(f"Request data: {request.json}")
        
        # Check if model is trained
        if trained_model is None:
            logger.error("Model not loaded")
            return jsonify({
                "error": "Model not trained. Please train the model first.",
                "status": "error"
            }), 400
        
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
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "status": "error"
                }), 400
        
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
            return jsonify({
                "error": f"Invalid data type: {str(e)}",
                "status": "error"
            }), 400
                
        # Encode user input if needed
        try:
            logger.debug("Encoding user input")
            encoded_user_input = trained_model.encode_user_input(user_data)
        except Exception as e:
            logger.error(f"Error encoding user input: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                "error": f"Error encoding user data: {str(e)}",
                "status": "error"
            }), 500
        
        # Add mock nutrition_needs if needed
        if not hasattr(trained_model, 'calculate_nutrition_needs'):
            logger.warning("Adding missing calculate_nutrition_needs method")
            
            def calculate_nutrition_needs(self, user_input):
                """
                Calculate daily caloric and macronutrient needs
                
                Args:
                    user_input (dict): User characteristics
                    
                Returns:
                    dict: Calculated nutrition needs
                """
                # Get user data
                age = user_input["age"]
                gender = user_input["gender"]
                weight_kg = user_input["weight"]
                height_cm = user_input["height"]
                activity_level = user_input["activity_level"]
                has_weight_gain = user_input["has_weight_gain"]
                has_weight_loss = user_input["has_weight_loss"]
                
                # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
                if gender == "Male":
                    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
                else:  # Female
                    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
                
                # Apply activity multiplier
                if isinstance(activity_level, int) and activity_level in self.activity_multipliers:
                    activity_multiplier = self.activity_multipliers[activity_level]
                else:
                    # Default to moderately active if invalid
                    activity_multiplier = 1.55
                
                # Calculate Total Daily Energy Expenditure (TDEE)
                tdee = bmr * activity_multiplier
                
                # Adjust calories based on weight goals
                if has_weight_gain:
                    daily_calories = tdee + 500  # Surplus for weight gain
                elif has_weight_loss:
                    daily_calories = max(1200, tdee - 500)  # Deficit for weight loss, min 1200 cal
                else:
                    daily_calories = tdee  # Maintenance
                
                # Calculate macronutrients
                # Protein: 0.8-1.2g per pound of body weight for weight maintenance
                # Higher for weight gain (1.0-1.5g), higher for weight loss (1.2-1.7g)
                weight_lbs = weight_kg * 2.20462  # Convert kg to lbs
                
                if has_weight_gain:
                    protein_g = weight_lbs * 1.2  # Higher protein for weight gain
                elif has_weight_loss:
                    protein_g = weight_lbs * 1.5  # Higher protein for weight loss
                else:
                    protein_g = weight_lbs * 1.0  # Moderate protein for maintenance
                
                # Fat: 20-35% of daily calories
                fat_percentage = 0.25  # 25% of calories from fat
                fat_calories = daily_calories * fat_percentage
                fat_g = fat_calories / 9  # 9 calories per gram of fat
                
                # Carbs: Remaining calories
                protein_calories = protein_g * 4  # 4 calories per gram of protein
                carb_calories = daily_calories - protein_calories - fat_calories
                carb_g = carb_calories / 4  # 4 calories per gram of carbs
                
                # Round values for cleaner display
                daily_calories = round(daily_calories)
                protein_g = round(protein_g)
                fat_g = round(fat_g)
                carb_g = round(carb_g)
                
                return {
                    "daily_calories": daily_calories,
                    "protein_g": protein_g,
                    "fat_g": fat_g,
                    "carb_g": carb_g
                }
            
            # Add the method to the model
            import types
            trained_model.calculate_nutrition_needs = types.MethodType(calculate_nutrition_needs, trained_model)
        
        # Add mock predict method if needed
        if not hasattr(trained_model, 'predict'):
            logger.warning("Adding missing predict method")
            
            def predict(self, user_input):
                """
                Predict meals for user input based on simplified logic
                """
                # Calculate nutrition needs
                nutrition_needs = self.calculate_nutrition_needs(user_input)
                
                # Default meal categories
                breakfast_options = ["oatmeal with berries", "greek yogurt with granola and fruit", "avocado toast with poached egg", 
                                     "vegetable omelet", "protein pancakes"]
                lunch_options = ["chicken and vegetable stir-fry", "quinoa salad with chickpeas and vegetables", 
                                 "tuna wrap with mixed greens", "lentil soup with whole grain bread", "turkey and avocado wrap"]
                dinner_options = ["grilled salmon with roasted vegetables", "vegetable stir-fry with brown rice", 
                                  "baked chicken with sweet potato", "tofu and vegetable stir-fry", "lean beef with broccoli"]
                snack_options = ["apple with almond butter", "greek yogurt with honey", "protein smoothie", 
                                "hummus with vegetable sticks", "mixed nuts"]
                
                # Generate meal recommendations
                breakfast_nutrition = []
                for breakfast in breakfast_options:
                    nutrition = self.get_meal_nutrition(breakfast, "breakfast")
                    breakfast_nutrition.append({
                        "meal": breakfast,
                        "nutrition": nutrition
                    })
                
                lunch_nutrition = []
                for lunch in lunch_options:
                    nutrition = self.get_meal_nutrition(lunch, "lunch")
                    lunch_nutrition.append({
                        "meal": lunch,
                        "nutrition": nutrition
                    })
                
                dinner_nutrition = []
                for dinner in dinner_options:
                    nutrition = self.get_meal_nutrition(dinner, "dinner")
                    dinner_nutrition.append({
                        "meal": dinner,
                        "nutrition": nutrition
                    })
                
                snack_nutrition = []
                for snack in snack_options:
                    nutrition = self.get_meal_nutrition(snack, "snack")
                    snack_nutrition.append({
                        "meal": snack,
                        "nutrition": nutrition
                    })
                
                return {
                    "nutrition_needs": nutrition_needs,
                    "breakfast_options": breakfast_nutrition,
                    "lunch_options": lunch_nutrition,
                    "dinner_options": dinner_nutrition,
                    "snack_options": snack_nutrition,
                    "model_accuracy": 0.95  # Placeholder
                }
            
            # Add the method to the model
            import types
            trained_model.predict = types.MethodType(predict, trained_model)
        
        # Get recommendations
        try:
            logger.debug("Getting recommendations")
            recommendations = trained_model.predict(encoded_user_input)
            logger.debug("Successfully generated recommendations")
        except Exception as e:
            logger.error(f"Error predicting recommendations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                "error": f"Error generating recommendations: {str(e)}",
                "status": "error"
            }), 500
        
        # Convert to serializable format
        try:
            logger.debug("Converting recommendations to serializable format")
            serializable_recommendations = convert_to_serializable(recommendations)
        except Exception as e:
            logger.error(f"Error serializing recommendations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                "error": f"Error processing results: {str(e)}",
                "status": "error"
            }), 500
        
        logger.info("Successfully processed prediction request")
        return jsonify({
            "status": "success",
            "recommendations": serializable_recommendations
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)