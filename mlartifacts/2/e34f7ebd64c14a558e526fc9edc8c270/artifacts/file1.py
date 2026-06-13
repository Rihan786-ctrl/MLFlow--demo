import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------
# MLflow Configuration
# -----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine_RF_Experiment")


# -----------------------------
# Load Dataset
# -----------------------------
wine = load_wine()

X = wine.data
y = wine.target


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)


# -----------------------------
# Hyperparameters
# -----------------------------
max_depth = 10
n_estimators = 10


# -----------------------------
# MLflow Run
# -----------------------------
with mlflow.start_run():

    # Train Model
    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    # Log Parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)

    #creating a confusing matrix plot and log it as an artifact
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted') 
    plt.ylabel('Actual')

    # Save the plot to a file
    plt.savefig("confusion_matrix.png")

    # Log the confusion matrix plot as an artifact
    mlflow.log_artifact("confusion_matrix.png")

    # Log File
    mlflow.log_artifact(__file__)

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )

    print(f"Accuracy: {accuracy:.4f}")

    