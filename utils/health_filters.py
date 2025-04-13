"""
Utility functions for filtering meal recommendations based on health conditions
"""

import logging

logger = logging.getLogger(__name__)

def apply_health_condition_filters(recommendations, user_data):
    """
    Filter and modify recommendations based on health conditions
    
    Args:
        recommendations (dict): Meal recommendations
        user_data (dict): User data including health conditions
        
    Returns:
        dict: Filtered recommendations
    """
    try:
        # First check if we have valid recommendations structure
        if not isinstance(recommendations, dict):
            return recommendations
            
        # Check health conditions from user data
        has_diabetes = user_data.get("has_diabetes", False)
        has_hypertension = user_data.get("has_hypertension", False)
        has_heart_disease = user_data.get("has_heart_disease", False)
        has_kidney_disease = user_data.get("has_kidney_disease", False)
        
        # Only continue if we have health conditions to consider
        if not any([has_diabetes, has_hypertension, has_heart_disease, has_kidney_disease]):
            return recommendations
            
        # Create filters based on health conditions
        avoid_keywords = []
        prefer_keywords = []
        
        # Add diabetes-related filters
        if has_diabetes:
            avoid_keywords.extend([
                'sugar', 'syrup', 'candy', 'cake', 'soda', 'white bread', 'pastry', 'donut', 
                'honey', 'jam', 'sweetened', 'dessert', 'ice cream', 'chocolate'
            ])
            prefer_keywords.extend([
                'whole grain', 'oats', 'quinoa', 'brown rice', 'lentil', 'bean', 
                'leafy green', 'vegetable', 'fiber', 'protein'
            ])
            
            # Add note about diabetes adjustments
            if "nutrition_needs" in recommendations:
                notes = recommendations["nutrition_needs"].get("notes", [])
                if not isinstance(notes, list):
                    notes = []
                notes.append("Prioritized low glycemic index foods and reduced added sugars")
                recommendations["nutrition_needs"]["notes"] = notes
        
        # Add hypertension-related filters
        if has_hypertension:
            avoid_keywords.extend([
                'salt', 'sodium', 'processed', 'canned', 'deli meat', 'bacon', 'sausage',
                'pickle', 'soy sauce', 'fast food', 'chips'
            ])
            prefer_keywords.extend([
                'potassium', 'magnesium', 'calcium', 'banana', 'spinach', 'kale', 
                'sweet potato', 'beans', 'lentils', 'fish', 'olive oil', 'garlic'
            ])
            
            # Add note about hypertension adjustments
            if "nutrition_needs" in recommendations:
                notes = recommendations["nutrition_needs"].get("notes", [])
                if not isinstance(notes, list):
                    notes = []
                notes.append("Prioritized low-sodium foods and potassium-rich options")
                recommendations["nutrition_needs"]["notes"] = notes
        
        # Add heart disease-related filters
        if has_heart_disease:
            avoid_keywords.extend([
                'saturated fat', 'trans fat', 'fried', 'fast food', 'processed meat',
                'full-fat dairy', 'butter', 'salt', 'sodium'
            ])
            prefer_keywords.extend([
                'omega-3', 'salmon', 'olive oil', 'avocado', 'nuts', 'fiber', 
                'whole grain', 'oats', 'fruits', 'vegetables', 'lean protein'
            ])
            
            # Add note about heart disease adjustments
            if "nutrition_needs" in recommendations:
                notes = recommendations["nutrition_needs"].get("notes", [])
                if not isinstance(notes, list):
                    notes = []
                notes.append("Prioritized heart-healthy foods and reduced saturated fats and sodium")
                recommendations["nutrition_needs"]["notes"] = notes
        
        # Add kidney disease-related filters
        if has_kidney_disease:
            avoid_keywords.extend([
                'phosphorus', 'potassium', 'banana', 'potato', 'tomato', 'avocado', 'dairy',
                'nuts', 'seeds', 'whole grain', 'brown rice', 'beans', 'lentils'
            ])
            prefer_keywords.extend([
                'rice milk', 'refined grain', 'white bread', 'apple', 'cucumber', 
                'green beans', 'lettuce', 'egg white', 'unsalted', 'low-sodium'
            ])
            
            # Add note about kidney disease adjustments
            if "nutrition_needs" in recommendations:
                notes = recommendations["nutrition_needs"].get("notes", [])
                if not isinstance(notes, list):
                    notes = []
                notes.append("Reduced foods high in phosphorus, potassium, and sodium")
                recommendations["nutrition_needs"]["notes"] = notes
        
        # Function to calculate health score for a meal
        def calculate_health_score(meal_name):
            if not meal_name or not isinstance(meal_name, str):
                return 0
                
            meal_lower = meal_name.lower()
            avoid_count = sum(1 for word in avoid_keywords if word in meal_lower)
            prefer_count = sum(1 for word in prefer_keywords if word in meal_lower)
            return prefer_count - (avoid_count * 1.5)  # Penalize avoided items more heavily
        
        # Process each meal type if they exist
        for meal_type in ["breakfast_options", "lunch_options", "dinner_options", "snack_options"]:
            if meal_type in recommendations:
                meal_options = recommendations[meal_type]
                
                # Safety check - make sure it's a list
                if not isinstance(meal_options, list):
                    continue
                
                # Score each meal option
                for meal_item in meal_options:
                    if isinstance(meal_item, dict) and "meal" in meal_item:
                        meal_item["health_score"] = calculate_health_score(meal_item["meal"])
                
                # Sort by health score (highest first)
                meal_options.sort(
                    key=lambda x: x.get("health_score", 0) if isinstance(x, dict) else 0, 
                    reverse=True
                )
                
                # Keep only the top 5 options
                recommendations[meal_type] = meal_options[:min(5, len(meal_options))]
        
        # Mark recommendations as health-adjusted
        recommendations["health_adjusted"] = True
        
        return recommendations
        
    except Exception as e:
        # Log the error but don't fail the whole request
        logger.error(f"Error applying health filters: {str(e)}")
        logger.error(traceback.format_exc())
        return recommendations  # Return original recommendations if filtering fails