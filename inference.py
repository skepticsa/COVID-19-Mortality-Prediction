
import os
import json
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the AutoGluon model"""
    from autogluon.tabular import TabularPredictor
    
    logger.info(f"Loading model from: {model_dir}")
    logger.info(f"Contents of model_dir: {os.listdir(model_dir)}")
    
    # IMPORTANT: Bypass Python version check
    predictor = TabularPredictor.load(model_dir, require_py_version_match=False)
    logger.info("Successfully loaded AutoGluon model (Python version check bypassed)")
    
    return predictor

def input_fn(request_body, content_type):
    """Parse input data"""
    logger.info(f"Content type: {content_type}")
    
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        df = pd.DataFrame(input_data)
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Input columns: {df.columns.tolist()}")
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    logger.info(f"Making predictions on data shape: {input_data.shape}")
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        if isinstance(prediction, pd.Series):
            return json.dumps(prediction.tolist())
        elif isinstance(prediction, np.ndarray):
            return json.dumps(prediction.tolist())
        else:
            return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {content_type}")
