from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

def train_baseline(X_train, y_train, class_weight=None):
    logreg = LogisticRegression(class_weight=class_weight)
    logreg.fit(X_train, y_train)

    return logreg

def train_gradient(X_train, y_train, scale_pos_weight=1):
    xgbc = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=67,
        eval_metric="logloss",
    )
    xgbc.fit(X_train, y_train)

    return xgbc

def train_random_forest(X_train, y_train, class_weight=None):
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight=class_weight,
        random_state=69
    )
    rf.fit(X_train, y_train)

    return rf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prauc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {"Precision Score" : precision,
            "Recall Score" : recall,
            "F1 Score" : f1,
            "PR-AUC Score" : prauc,
            "Confusion Matrix" : cm}