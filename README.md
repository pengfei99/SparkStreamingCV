# SparkStreamingCV


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
