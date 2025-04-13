"""
Serialization utilities for converting data to JSON-compatible formats
"""

import numpy as np
import pandas as pd

def convert_to_serializable(obj):
    """
    Convert object to JSON serializable format
    
    Args:
        obj: Object to convert
            
    Returns:
        JSON serializable object
    """
    if obj is None:
        return None
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
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