from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def train_baseline_models(X, Y, test_size=0.2, random_state=42):
    """Train Random Forest and XGBoost baseline models"""
    
    # Tokenize and vectorize X variable (texts)
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_vectorized, Y, test_size=test_size, random_state=random_state
    )
    
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Train models
    print("Training Random Forest...")
    rf_model.fit(X_train, Y_train)
    
    print("Training XGBoost...")
    xgb_model.fit(X_train, Y_train)
    
    # Get predictions
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    
    # Print results
    print("\n=== Random Forest Results ===")
    print(classification_report(Y_test, rf_preds, digits=4, target_names=['Non-Spam', 'Spam']))
    
    print("\n=== XGBoost Results ===")
    print(classification_report(Y_test, xgb_preds, digits=4, target_names=['Non-Spam', 'Spam']))
    
    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'vectorizer': vectorizer,
        'test_data': (X_test, Y_test),
        'predictions': {'rf': rf_preds, 'xgb': xgb_preds}
    }

def save_baseline_models(models_dict, save_dir='models/'):
    """Save trained baseline models"""
    with open(f'{save_dir}rf_model.pkl', 'wb') as f:
        pickle.dump(models_dict['rf_model'], f)
    
    with open(f'{save_dir}xgb_model.pkl', 'wb') as f:
        pickle.dump(models_dict['xgb_model'], f)
    
    with open(f'{save_dir}vectorizer.pkl', 'wb') as f:
        pickle.dump(models_dict['vectorizer'], f)
    
    print(f"Models saved to {save_dir}")

def load_baseline_models(save_dir='models/'):
    """Load saved baseline models"""
    with open(f'{save_dir}rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(f'{save_dir}xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open(f'{save_dir}vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return rf_model, xgb_model, vectorizer