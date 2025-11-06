# src/predict.py
import pandas as pd
import numpy as np
import joblib
import os
from util.commonUtil import get_data_source
from util.logUtil import get_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
LOG_DIR = os.path.join(BASE_DIR, 'log')

logger = get_logger('predict', log_dir=LOG_DIR)


def load_model():
    model_path = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
    preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')
    poly_path = os.path.join(MODEL_DIR, 'poly.pkl')
    threshold_path = os.path.join(MODEL_DIR, 'threshold.txt')

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    poly = joblib.load(poly_path)
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())

    return model, preprocessor, poly, threshold


def predict(test_path):
    model, preprocessor, poly, threshold = load_model()

    test_df = get_data_source(test_path)
    X_test = test_df.copy()

    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours']
    X_test = X_test.drop(columns=drop_cols, errors='ignore')

    # 特征工程（与训练一致）
    def ultimate_features(df):
        df = df.copy()
        for col in ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        df['IncomePerAge'] = df['MonthlyIncome'] / df['Age']
        df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
        df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
        df['LowPayHighOT'] = (df['MonthlyIncome'] < 3000) & (df['OverTime'] == 'Yes')
        df['RecentHire'] = df['YearsAtCompany'] < 1
        return df

    X_test = ultimate_features(X_test)

    cat_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    interaction_cols = ['MonthlyIncome', 'TotalWorkingYears', 'Age', 'DistanceFromHome']

    poly_features = poly.transform(X_test[interaction_cols])
    X_test = X_test.drop(columns=interaction_cols)

    X_main = preprocessor.transform(X_test)
    X_processed = np.hstack([X_main, poly_features])

    prob = model.predict_proba(X_processed)[:, 1]
    pred = (prob >= threshold).astype(int)

    # 后处理
    high_risk = (test_df['OverTime'] == 'Yes') & (test_df['MonthlyIncome'] < 3000) & (test_df['YearsAtCompany'] < 2)
    pred[high_risk] = 1

    return pred


if __name__ == "__main__":
    test_path = os.path.join(DATA_DIR, 'test.csv')
    pred = predict(test_path)
    submission = pd.DataFrame({'Attrition': pred})
    submission.to_csv(os.path.join(DATA_DIR, 'submission_predict.csv'), index=False)
    logger.info("Prediction completed.")