#! /bin/bash
unset AWS_SESSION_TOKEN
export MLFLOW_TRACKING_URI='https://mlflow.lab.sspcloud.fr/'

export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export AWS_ACCESS_KEY_ID=mlflow
export AWS_SECRET_ACCESS_KEY=changeMe
export AWS_DEFAULT_REGION=us-east-1
poetry run python model_training/FetchModelFromMLFlow.py