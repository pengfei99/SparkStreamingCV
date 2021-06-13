FROM inseefrlab/jupyter-datascience:master

ENV MODEL_HOME="/opt/models"
RUN mkdir -p  ${MODEL_HOME}
RUN pip uninstall statsmodels -y && \
    pip install statsmodels==0.11.0 && \
    pip install opencv-contrib-python && \
    pip install tensorflow && \
    pip install mlflow[extras]==1.14.1 && \
    pip install --upgrade protobuf