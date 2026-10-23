"""
Модуль инференса — загружает модель и классифицирует потоки
classifier.py
"""

import os
import numpy as np
import joblib


class Classifier:
    """
    Загружает обученную модель Random Forest и scaler_app.pkl.
    scaler_app.pkl обучен на сырых признаках трафика (байты, мкс) —
    именно такие значения поступают из capture.py.
    """

    def __init__(self, model_path: str, scaler_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler не найден: {scaler_path}")

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"[Classifier] Модель загружена:  {model_path}")
        print(f"[Classifier] Scaler загружен:   {scaler_path}")

    def predict(self, features: np.ndarray) -> dict:
        """
        Принимает вектор сырых признаков shape (10,):
          [Bwd Packet Length Std, Average Packet Size,
           Packet Length Variance, Total Length of Bwd Packets,
           Fwd IAT Std (мкс), Total Length of Fwd Packets,
           Fwd Packet Length Max, Fwd Packet Length Mean,
           Flow IAT Max (мкс), Fwd Header Length]

        Нормализует через scaler_app.pkl и передаёт в модель.
        """
        X        = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        label = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0]

        return {
            "label":       label,
            "label_str":   "ATTACK" if label == 1 else "BENIGN",
            "prob_attack": float(proba[1]) * 100,
            "confidence":  float(max(proba)) * 100,
        }
