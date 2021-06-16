# Face mask detection
This project aims to develop an application which can detect human faces and check if they wear a mask or not in an 
image or a video stream.

## Project summary
This project first extract human faces by using Haar-Cascades Classifier, because one image may contain many faces.
For each extracted face, the application will use a pre-trained convolutional neural network (CNN) Classifier to predict
if they have face mask or not. Figure-1 shows the architecture of the application
![app architecture](https://minio.lab.sspcloud.fr/pengfei/diffusion/face-mask-detection/face-detection-architecture.png)
**Figure-1** Face mask detection application architecture

Figure-2 shows an example of the data pipeline
![data pipeline](https://minio.lab.sspcloud.fr/pengfei/diffusion/face-mask-detection/Face_detection_pipeline.png)
**Figure-2** Face mask detection application data pipeline

 ## SSP Datalab
This project is originally developed as a pure python project without any ML Ops supports. It can only be deployed on a 
single machine. When we apply this application on a large image dataset or a video stream, we have encountered serious
performance issue. 

We want to highlight two major improvements which Datalab provides us. 
1. Data lab provides a Ml Ops service, which allow us to perform CI, CD, and CT easily. 
2. By converting the project to use Spark (a unified analytics engine, built for big data processing), we resolve the 
   performance issue. Datalab provides built-in spark support, which allow us to deploy a spark cluster(up to 100 nodes)
   to run the application


### ML Ops service inside Datalab
Copy?

### Parallel computation inside Datalab
Datalab provides many parallel computation framework such as Spark. When a user creates a spark session, he can specify 
the size of his spark cluster. 

For example, with the following command, I will create a spark cluster which contains five nodes.

```python
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder.master("k8s://https://kubernetes.default.svc:443") \
    .appName("SparkStreamingComputerVision") \
    .config("spark.kubernetes.container.image", "inseefrlab/jupyter-datascience:master") \
    .config("spark.executor.instances", "5") \
    .config("spark.eventLog.enabled","true") \
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") \
    .config("spark.archives", "pyspark_conda_env.tar.gz#environment") \
    .getOrCreate()
```

# Data and model

The dataset for training the cnn model can be found [here](https://www.kaggle.com/nageshsingh/mask-and-social-distancing-detection-using-vgg19)

The dataset for testing the application can be found [here](https://www.kaggle.com/andrewmvd/face-mask-detection)

The source codes of the application can be found [here](https://github.com/pengfei99/SparkStreamingCV.git)








# SparkStreamingCV

## 1. Create jupyter kernel 

### 1.1 Create kernel by using poetry venv

```shell
poetry run python -m ipykernel install --name spark_streaming_cv --user 
```

### 1.2 Create kernel by using conda venv

```shell
python -m ipykernel install --name spark_streaming_cv --user 
```


## Setup conda env and install dependencies
```shell
conda create -y -n ${venv_name} python=${py_version}
conda init
source $HOME/.bashrc
conda activate ${venv_name}
pip install -r freshpy38-requirements.txt
```

## pack conda virtual env 
```shell
conda pack -f -o pyspark_conda_env.tar.gz
```
