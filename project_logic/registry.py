import os
import pickle

MODEL_PATH = 'models/best_pipeline_bin_death_score.pkl'

def save_model(pipeline):
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")

def load_model_trained():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run training first.")
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
    return pipeline
