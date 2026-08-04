import os
import joblib
import json
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class CreditRiskPipeline:
    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            # Dynamically resolve relative to src/ to point to ../models/artifacts
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.artifacts_dir = os.path.join(base_dir, "..", "models", "artifacts")
        else:
            self.artifacts_dir = artifacts_dir
            
        self.load_artifacts()

    def load_artifacts(self):
        """Loads all required models, scalers, and configs from the artifacts directory."""
        # Configs & Meta
        self.preprocessing_config = joblib.load(f"{self.artifacts_dir}/preprocessing_config.pkl")
        self.feature_engineering = joblib.load(f"{self.artifacts_dir}/feature_engineering.pkl")
        self.model_columns = joblib.load(f"{self.artifacts_dir}/model_columns.pkl")
        
        with open(f"{self.artifacts_dir}/threshold_info.json", "r") as f:
            self.threshold_info = json.load(f)
            self.optimal_threshold = self.threshold_info["optimal_threshold"]

        # Scaler
        self.scaler = joblib.load(f"{self.artifacts_dir}/scaler.pkl")

        # Models
        self.model_cat = joblib.load(f"{self.artifacts_dir}/calibrated_cat.pkl")
        self.model_xgb = joblib.load(f"{self.artifacts_dir}/calibrated_xgb.pkl")
        self.model_lgb = joblib.load(f"{self.artifacts_dir}/calibrated_lgb.pkl")
        
        # Explainer (Optional, for SHAP)
        try:
            self.shap_explainer = joblib.load(f"{self.artifacts_dir}/shap_explainer.pkl")
        except FileNotFoundError:
            self.shap_explainer = None

    def preprocess_input(self, input_dict):
        """
        Takes a raw dictionary of inputs and transforms it into the exact format 
        required by the models.
        """
        # 1. Convert input to DataFrame
        df = pd.DataFrame([input_dict])

        # Fill in missing columns with 0 that are required for intermediate steps (scaling) but dropped later.
        for col in self.preprocessing_config["scaling_cols"]:
            if col not in df.columns:
                df[col] = 0.0

        # 2. Handle missing values & replacements based on training data
        if "residence_type" in df.columns:
            df["residence_type"] = df["residence_type"].fillna(
                self.preprocessing_config["imputation"]["residence_type_mode"]
            )
            
        if "loan_purpose" in df.columns:
            df["loan_purpose"] = df["loan_purpose"].replace(
                self.preprocessing_config["replacements"]["loan_purpose"]
            )

        # Logical capping from notebook
        if "years_at_current_address" in df.columns and "age" in df.columns:
            df["years_at_current_address"] = np.minimum(df["years_at_current_address"], df["age"])

        # 3. Feature Engineering (safe calculation to avoid division by zero)
        df["loan_to_income"] = np.where(
            df["income"] != 0, 
            df["loan_amount"] / df["income"], 
            0
        )
        
        df["dpd_rate"] = np.where(
            df["total_loan_months"] != 0,
            (df["total_dpd"] * 100) / df["total_loan_months"],
            0
        )
        
        df["delinquency_rate"] = np.where(
            df["total_loan_months"] != 0,
            (df["delinquent_months"] * 100) / df["total_loan_months"],
            0
        )
        
        df["avg_dpd_per_delinquent"] = np.where(
            df["delinquent_months"] != 0,
            (df["total_dpd"] * 100) / df["delinquent_months"],
            0
        )
        
        # 4. Log Transformations
        cols_to_log = self.preprocessing_config["cols_to_log"]
        for col in cols_to_log:
            if col in df.columns:
                df[col] = np.log1p(df[col])
                
        # 5. Standard Scaling (Must happen BEFORE dropping cols_to_drop)
        scaling_cols = self.preprocessing_config["scaling_cols"]
        df[scaling_cols] = self.scaler.transform(df[scaling_cols])

        # 6. Drop Unnecessary Columns
        cols_to_drop = self.preprocessing_config["cols_to_drop"]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        # 7. One-Hot Encoding (Get Dummies)
        df = pd.get_dummies(df, drop_first=True)
        
        # Ensure boolean columns are integers if any
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)

        # 8. Align columns exactly to the trained model features
        df = df.reindex(columns=self.model_columns, fill_value=0)

        return df

    def _robust_predict_proba(self, calibrated_model, df_processed):
        """
        Bypasses Scikit-Learn 1.6+ validation checks for unpickled models 
        to fix the XGBoost Regressor tag mismatch error.
        """
        try:
            return calibrated_model.predict_proba(df_processed)[:, 1]
        except Exception:
            probs = []
            for cc in calibrated_model.calibrated_classifiers_:
                base_probs = cc.estimator.predict_proba(df_processed)[:, 1]
                calib_prob = cc.calibrators[0].predict(base_probs)
                probs.append(calib_prob)
            return np.mean(probs, axis=0)

    def predict(self, df_processed):
        """
        Runs the ensemble prediction and applies the threshold.
        """
        # Get probability of Default (class 1) from each model robustly
        prob_cat = self._robust_predict_proba(self.model_cat, df_processed)
        prob_xgb = self._robust_predict_proba(self.model_xgb, df_processed)
        prob_lgb = self._robust_predict_proba(self.model_lgb, df_processed)

        # Master Ensemble Average
        master_prob = (prob_cat + prob_xgb + prob_lgb) / 3.0
        final_prob = master_prob[0]
        
        # Compare to optimal threshold
        is_default = int(final_prob >= self.optimal_threshold)
        
        return {
            "prediction": is_default, 
            "probability": final_prob,
            "threshold": self.optimal_threshold,
            "models_breakdown": {
                "CatBoost": prob_cat[0],
                "XGBoost": prob_xgb[0],
                "LightGBM": prob_lgb[0]
            }
        }
