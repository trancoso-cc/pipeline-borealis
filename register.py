import mlflow, mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
REGISTERED_URI = "models:/m-8842/5"                 # provides: model-uri
ARTIFACT_PATH  = "gs://acme-ml-reg-9931/m-8842/"  # provides: gcs artifact path
mlflow.sklearn.log_model(GradientBoostingClassifier().fit(X, y), "model",
                         registered_model_name="m-8842")
