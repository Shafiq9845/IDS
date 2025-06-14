import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load models and preprocessing objects
scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")
rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")
dnn_model = load_model("models/dnn_model.keras")
label_encoder = joblib.load("models/label_encoder.pkl")
all_feature_names = list(scaler.feature_names_in_)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", feature_names=all_feature_names)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "csv_file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    file = request.files["csv_file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    df = pd.read_csv(file)
    df.columns = [col.strip().lower().replace('_', ' ').replace('-', ' ').replace('.', '.').replace('/', '/').replace('  ', ' ') for col in df.columns]
    df.columns = [col.replace('  ', ' ') for col in df.columns]
    for col in all_feature_names:
        if col not in df.columns:
            df[col] = 0

    # Add this block:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # filepath: c:\Users\Shafiq\Documents\Reserch_Colab\app.py
    missing_cols = [col for col in all_feature_names if col not in df.columns]
    if missing_cols:
        flash(f"Missing columns in uploaded file: {missing_cols}")
        return redirect(url_for("index"))
    X = df[all_feature_names]
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    # Predict with all models
    rf_preds = rf_model.predict(X_selected)
    xgb_preds = xgb_model.predict(X_selected)
    dnn_probs = dnn_model.predict(X_selected)
    dnn_preds = np.argmax(dnn_probs, axis=1)

    # Decode labels if needed
    rf_labels = label_encoder.inverse_transform(rf_preds)
    xgb_labels = label_encoder.inverse_transform(xgb_preds)
    dnn_labels = label_encoder.inverse_transform(dnn_preds)

    # Add predictions to DataFrame
    df["RandomForest_Pred"] = rf_labels
    df["XGBoost_Pred"] = xgb_labels
    df["DNN_Pred"] = dnn_labels

    # If ground truth is present, calculate metrics
    metrics = {}
    if "label" in df.columns:
        y_true = label_encoder.transform(df["label"])
        labels = np.arange(len(label_encoder.classes_))
        # Calculate accuracy for each model
        metrics["rf_accuracy"] = accuracy_score(y_true, rf_preds)
        metrics["xgb_accuracy"] = accuracy_score(y_true, xgb_preds)
        metrics["dnn_accuracy"] = accuracy_score(y_true, dnn_preds)
        metrics["rf_report"] = classification_report(
            y_true, rf_preds, labels=labels, target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )
        metrics["xgb_report"] = classification_report(
            y_true, xgb_preds, labels=labels, target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )
        metrics["dnn_report"] = classification_report(
            y_true, dnn_preds, labels=labels, target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )

        # Confusion matrix for XGBoost
        cm = confusion_matrix(y_true, xgb_preds, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('XGBoost Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join("static", "xgboost_confusion_matrix.png")
        os.makedirs("static", exist_ok=True)
        plt.savefig(cm_path)
        plt.close()
        metrics["xgb_cm_path"] = cm_path

    actual_counts = df['label'].value_counts().to_dict() if 'label' in df.columns else {}
    rf_counts = df['RandomForest_Pred'].value_counts().to_dict()
    xgb_counts = df['XGBoost_Pred'].value_counts().to_dict()
    dnn_counts = df['DNN_Pred'].value_counts().to_dict()

    all_labels = sorted(set(list(actual_counts.keys()) + list(rf_counts.keys()) + list(xgb_counts.keys()) + list(dnn_counts.keys())))

    return render_template(
        "results.html",
        tables=[df.to_html(classes='data')],
        titles=df.columns.values,
        metrics=metrics,
        df=df,
        actual_counts=actual_counts,
        rf_counts=rf_counts,
        xgb_counts=xgb_counts,
        dnn_counts=dnn_counts,
        all_labels=all_labels
    )

if __name__ == "__main__":
    app.run(debug=True)