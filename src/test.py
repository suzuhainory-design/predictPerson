# src/test.py
import os
import joblib
from src.train import *


# 在 train.py 训练完成后运行
def save_model_artifacts():
    joblib.dump(ensemble, os.path.join(MODEL_DIR, 'ensemble_model.pkl'))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    joblib.dump(poly, os.path.join(MODEL_DIR, 'poly.pkl'))
    with open(os.path.join(MODEL_DIR, 'threshold.txt'), 'w') as f:
        f.write(str(best_thr))
    logger.info("Model artifacts saved to model/.")


if __name__ == "__main__":
    save_model_artifacts()