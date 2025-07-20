import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import shap
import numpy as np
import matplotlib.pyplot as plt

class FraudDetector:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.scaler = StandardScaler()

    def preprocess(self):
        df = self.df.copy()
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
        df['TimeDiff'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds()

        df['TransactionType'] = df['TransactionType'].astype('category').cat.codes
        df['Channel'] = df['Channel'].astype('category').cat.codes
        df['Location'] = df['Location'].astype('category').cat.codes
        df['DeviceID'] = df['DeviceID'].astype('category').cat.codes
        df['MerchantID'] = df['MerchantID'].astype('category').cat.codes
        df['CustomerOccupation'] = df['CustomerOccupation'].astype('category').cat.codes

        self.feature_cols = [
            'TransactionAmount', 'TransactionType', 'Location', 'Channel', 'DeviceID',
            'MerchantID', 'CustomerAge', 'CustomerOccupation', 'TransactionDuration',
            'LoginAttempts', 'AccountBalance', 'TimeDiff'
        ]
        df = df[self.feature_cols].fillna(0)
        self.X_raw = df  # Save raw for explanation
        scaled = self.scaler.fit_transform(df)
        return scaled

    def generate_explanations(self, X, top_n=3):
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)

        friendly_names = {
            "TransactionAmount": "Transaction amount was high",
            "LoginAttempts": "Unusual number of login attempts",
            "AccountBalance": "Account balance was low",
            "TransactionDuration": "Transaction took unusually long",
            "TimeDiff": "Short time since previous transaction",
            "CustomerAge": "Customer age was uncommon for such activity",
            "Channel": "Unusual channel used",
            "Location": "Suspicious location detected",
            "CustomerOccupation": "Uncommon occupation for this transaction type",
            "DeviceID": "Unfamiliar device used",
            "MerchantID": "Unrecognized merchant",
            "TransactionType": "Unusual transaction type"
        }

        risk_flags = {
            "TransactionAmount": lambda val: "very high" if val > 100000 else "high" if val > 50000 else "moderate",
            "AccountBalance": lambda val: "very low" if val < 1000 else "low" if val < 5000 else "sufficient",
            "LoginAttempts": lambda val: "multiple failed" if val >= 3 else "normal",
            "TimeDiff": lambda val: "immediately after previous" if val < 60 else "short interval" if val < 300 else "normal delay"
        }

        explanations = []

        for i in range(len(X)):
            row_values = shap_values[i].values
            row_data = self.X_raw.iloc[i]
            top_indices = abs(row_values).argsort()[::-1][:top_n]

            bullets = []

            for idx in top_indices:
                feature = self.feature_cols[idx]
                val = row_data[feature]
                impact = row_values[idx]
                direction = "increased" if impact > 0 else "reduced"

                description = friendly_names.get(feature, f"{feature} was unusual")
                risk_level = ""

                if feature in risk_flags:
                    flag = risk_flags[feature](val)
                    risk_level = f" ({flag})"

                bullets.append(f"<li style='color:#e63946'><b>{description}{risk_level}</b> → {direction} fraud risk.</li>")

            detailed_text = """
            <div style='color:#333;'>
            <p><strong>The model flagged this transaction as fraudulent based on the following factors:</strong></p>
            <ul>
            """ + "\n".join(bullets) + "\n</ul></div>"
            explanations.append(detailed_text)

        return explanations

    def plot_shap_waterfall(self, X, shap_values, index, feature_names):
        shap_values.feature_names = feature_names

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[index], max_display=8, show=False)
        return fig

    def predict(self):
        X_scaled = self.preprocess()
        self.model.fit(X_scaled)
        scores = self.model.decision_function(X_scaled)
        preds = self.model.predict(X_scaled)

        self.df['AnomalyScore'] = preds
        self.df['IsFraud'] = self.df['AnomalyScore'].apply(lambda x: 1 if x == -1 else 0)

        score_min = scores.min()
        score_max = scores.max()
        self.df['FraudProbability'] = 1 - (scores - score_min) / (score_max - score_min)

        fraud_indices = self.df[self.df['IsFraud'] == 1].index.tolist()
        if fraud_indices:
            X_fraud = self.X_raw.iloc[fraud_indices].fillna(0)
            X_fraud_scaled = self.scaler.transform(X_fraud)
            explanations = self.generate_explanations(X_fraud_scaled)
            self.df['FraudExplanation'] = ""
            self.df.loc[fraud_indices, 'FraudExplanation'] = explanations
        else:
            self.df['FraudExplanation'] = ""

        return self.df
