#! /bin/bash
unset AWS_SESSION_TOKEN
export MLFLOW_TRACKING_URI=''
export MLFLOW_EXPERIMENT_NAME="mask-detection"
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=changeMe
export AWS_DEFAULT_REGION=us-east-1
poetry run python model_training/TrainVgg19FaceMask.py