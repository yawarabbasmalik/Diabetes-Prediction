from importing_lib import *
import os
import pickle

# ── Function 1: Load Processed Data ─────────────────────────────
def load_processed_data():
    processed_path = os.path.join(os.path.dirname(__file__), '..', 'processed')
    
    train_data = pd.read_csv(os.path.join(processed_path, 'train_processed.csv'))
    test_data  = pd.read_csv(os.path.join(processed_path, 'test_processed.csv'))
    
    print(f"Train data loaded : {train_data.shape}")
    print(f"Test data loaded  : {test_data.shape}")
    print("************************")
    
    return train_data, test_data

# ── Function 2: Split Features & Target ─────────────────────────
def split_features_target(train_data, test_data):
    x_train = train_data.iloc[:, :-1]   # all columns except last
    y_train = train_data.iloc[:, -1:]   # last column (Outcome)
    
    x_test  = test_data.iloc[:, :-1]
    y_test  = test_data.iloc[:, -1:]
    
    print(f"x_train shape : {x_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"x_test shape  : {x_test.shape}")
    print(f"y_test shape  : {y_test.shape}")
    print("************************")
    
    return x_train, x_test, y_train, y_test

# ── Function 3: Train Model ──────────────────────────────────────
def train_model(x_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    
    model = GaussianNB()
    model.fit(x_train, y_train.values.ravel())
    
    print("Model training completed!")
    print("************************")
    
    return model

# ── Function 4: Evaluate Model ───────────────────────────────────
def evaluate_model(model, x_test, y_test):
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score,
                                 classification_report, confusion_matrix)
    
    predicted = model.predict(x_test)
    
    print("========== Model Evaluation ==========")
    print(f"Accuracy  : {accuracy_score(y_test, predicted)  * 100:.2f} %")
    print(f"Precision : {precision_score(y_test, predicted) * 100:.2f} %")
    print(f"Recall    : {recall_score(y_test, predicted)    * 100:.2f} %")
    print(f"F1 Score  : {f1_score(y_test, predicted)        * 100:.2f} %")
    print("************************")
    print("Classification Report:")
    print(classification_report(y_test, predicted))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predicted))
    print("************************")

# ── Function 5: Save Model with Pickle ──────────────────────────
def save_model(model):
    # Why pickle? It serializes (saves) the entire trained model
    # to a file so you can reload and reuse it later without
    # retraining — very useful for deployment & predictions

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_path, exist_ok=True)
    
    model_file = os.path.join(model_path, 'naive_bayes_model.pkl')
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_file}")
    print("************************")

# ── Function 6: Load Model with Pickle ──────────────────────────
def load_model():
    model_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'naive_bayes_model.pkl')
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully!")
    return model

# ── Main Function ────────────────────────────────────────────────
def main():
    print("========== STEP 1: Loading Processed Data ==========")
    train_data, test_data = load_processed_data()
    
    print("========== STEP 2: Splitting Features & Target ==========")
    x_train, x_test, y_train, y_test = split_features_target(train_data, test_data)
    
    print("========== STEP 3: Training Model ==========")
    model = train_model(x_train, y_train)
    
    print("========== STEP 4: Evaluating Model ==========")
    evaluate_model(model, x_test, y_test)
    
    print("========== STEP 5: Saving Model ==========")
    save_model(model)
    
    return model

# ── Call Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    model = main()

import json

def evaluate_model(model, x_test, y_test):
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score,
                                 classification_report, confusion_matrix)

    predicted = model.predict(x_test)

    accuracy  = accuracy_score(y_test, predicted)  * 100
    precision = precision_score(y_test, predicted) * 100
    recall    = recall_score(y_test, predicted)    * 100
    f1        = f1_score(y_test, predicted)        * 100

    print(f"Accuracy  : {accuracy:.2f} %")
    print(f"Precision : {precision:.2f} %")
    print(f"Recall    : {recall:.2f} %")
    print(f"F1 Score  : {f1:.2f} %")

    # ── Save metrics for DVC tracking ───────────────────────────
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'metrics')
    os.makedirs(metrics_path, exist_ok=True)

    scores = {
        "accuracy" : round(accuracy,  2),
        "precision": round(precision, 2),
        "recall"   : round(recall,    2),
        "f1_score" : round(f1,        2)
    }

    with open(os.path.join(metrics_path, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)

    print("Metrics saved!")
    print("************************")