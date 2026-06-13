import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run():
    mlflow.log_param("test", 123)
    mlflow.log_metric("accuracy", 0.99)