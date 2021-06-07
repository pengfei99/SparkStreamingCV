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
pip install -r requirements.txt
```

## pack conda virtual env 
```shell
conda pack -f -o pyspark_conda_env.tar.gz
```
