import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class LightPricePredictor:
    def __init__(self, predict_days=3, test_size=0.2):
        self.predict_days = predict_days
        self.test_size = test_size
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def create_features(self, df):
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        for window in [3, 5, 10]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'close_ma_{window}'] = df['close'] / df[f'ma_{window}'] - 1
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
        
        for i in range(1, 4):
            df[f'high_{i}'] = df['high'].shift(i)
            df[f'low_{i}'] = df['low'].shift(i)
        
        return df
    
    def create_target(self, df):
        df = df.copy()
        future_return = df['close'].shift(-self.predict_days) / df['close'] - 1
        df['target'] = (future_return > 0.005).astype(int)
        return df
    
    def prepare_data(self, df):
        df = self.create_features(df)
        df = self.create_target(df)
        df = df.dropna()
        
        exclude_cols = ['date', 'target', 'open', 'high', 'low', 'close', 'volume']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        train_size = int(len(df) * (1 - self.test_size))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        X_train = train_df[self.feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[self.feature_cols].values
        y_test = test_df['target'].values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_df
    
    def train(self, df):
        X_train, X_test, y_train, y_test, test_df = self.prepare_data(df)
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, test_pred))
        
        return test_df, test_pred
    
    def predict_next(self, df):
        df_features = self.create_features(df)
        df_features = df_features.dropna()
        
        last_data = df_features.iloc[-1][self.feature_cols].values.reshape(1, -1)
        last_data_scaled = self.scaler.transform(last_data)
        
        prob = self.model.predict_proba(last_data_scaled)[0]
        prediction = 1 if prob[1] > 0.5 else 0
        
        return {
            'prediction': 'UP' if prediction == 1 else 'NOT UP',
            'up_probability': float(prob[1]),
            'confidence': float(max(prob)),
            'predict_days': self.predict_days
        }
    
    def get_feature_importance(self, top_n=10):
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance.head(top_n)

def main():
    print("=" * 50)
    print("Lightweight Short-term Price Predictor")
    print("=" * 50)
    
    df = pd.read_csv('000852.SH.csv')
    print(f"\nData loaded: {len(df)} records")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    
    predictor = LightPricePredictor(predict_days=3, test_size=0.2)
    
    print(f"\nTraining model (predicting next {predictor.predict_days} days)...")
    test_df, test_pred = predictor.train(df)
    
    print("\n" + "=" * 50)
    print("Top 10 Feature Importance:")
    print("=" * 50)
    print(predictor.get_feature_importance(10))
    
    print("\n" + "=" * 50)
    print("Latest Prediction:")
    print("=" * 50)
    result = predictor.predict_next(df)
    print(f"Prediction Direction: {result['prediction']}")
    print(f"Up Probability: {result['up_probability']:.2%}")
    print(f"Model Confidence: {result['confidence']:.2%}")
    print(f"Prediction Period: {result['predict_days']} trading days")
    
    print("\n" + "=" * 50)
    print("Backtest Signals (Last 20 Trading Days):")
    print("=" * 50)
    
    test_df['prediction'] = test_pred
    recent = test_df.tail(20)[['date', 'close', 'target', 'prediction']].copy()
    recent['correct'] = recent['target'] == recent['prediction']
    print(recent.to_string(index=False))
    
    print(f"\nRecent Prediction Accuracy: {(recent['correct'].sum() / len(recent)):.2%}")

if __name__ == "__main__":
    main()
