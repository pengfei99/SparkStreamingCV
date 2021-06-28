# Spark on k8s use pod template 

Before spark 3.0, it's difficult to configure driver, executor running env by configuring pod specification. 
After Spark 3.0, we can set custom Pod configuration for executor/driver through PodTemplate. 

The PodTemplate is a YAML file that describes the metadata/spec fields of the pod which runs spark Driver/Executor.

To achieve this, you need to do two things:
1. Edit one PodTemplate yaml file
2. Include this PodTemplate file into your spark context.

## PodTemplate yaml file

The PodTemplate file is like any k8s Pod configuration file. Just one exception, you can't have multiple containers 
inside the Pod. But you can have multiple init containers inside.

Note that init container has two point which is different from normal container
1. Init containers always run to completion.
2. Each init container must complete successfully before the next one starts.

Below is an example of how we add init container to spark driver/executor pod

```yaml
apiVersion: v1
kind: Pod
metadata:
 name: rss-site
 labels:
   app: web
spec:
  initContainers:
    - name: init-sleep
      image: busybox
      command: ['sh','-c','sleep 1']
    - name: init-echo
      image: busybox
      comand: ['sh','-c','echo "pod template "']
    
  containers:
  - name: 
       image: inseefrlab/jupyter-datascience:master
       command: ['sh', '-c', '/opt/entrypoint.sh executor']


```

You may wonder why we need this pod template. We have many use cases:
1. The default image of the spark executor may not contain the dependencies that your application need.
For scala/java application, it's easy, you can use .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0")
But for python, it's more complicate.
2. Add conf to spark executor/driver that the sparkConf does not support (e.g. NodeSelector/Toleration). 

Below is an PodTemplate example that we add python tensorflow dependencies to a spark executor/driver that runs
   on datalab k8s

```yaml
apiversion: v1
kind: Pod
spec:
  containers:
  - name: custom_py_dependencies
    image: inseefrlab/jupyter-datascience:master
    command: ['sh', '-c', '/opt/conda/bin/pip install tensorflow && /opt/entrypoint.sh executor']
```

## Include podTemplate into your spark Context

### Mode spark submit
In mode spark submit you can add the podTemplate by using 
- spark.kubernetes.executor.podTemplateFile
- spark.kubernetes.driver.podTemplateFile

Below is a complete example
```shell
opt/spark/bin/spark-submit
--deploy-mode=cluster
--class org.apache.spark.examples.SparkPi
--master=k8s://https://172.17.0.1:443
--conf spark.kubernetes.namespace=${KUBERNETES_NAMESPACE}
--conf spark.kubernetes.driver.container.image=liupengfei99/sparkstreamincv:latest
--conf spark.kubernetes.executor.container.image=liupengfei99/sparkstreamincv:latest
--conf spark.driver.cores=1
--conf spark.driver.memory=4096M
--conf spark.executor.cores=1
--conf spark.executor.memory=4096M
--conf spark.executor.instances=2
--conf spark.kubernetes.driver.podTemplateFile=/path/to/pod_template.yaml
--conf spark.kubernetes.executor.podTemplateFile=/path/to/pod_template.yaml
--conf spark.kubernetes.executor.deleteOnTermination=false
local:///opt/spark/examples/jars/spark-examples_2.12-3.0.0-SNAPSHOT.jar 100

```

### Mode spark-shell

```shell
pyspark --master=k8s://https://kubernetes.default.svc:443 
--conf spark.kubernetes.namespace=${KUBERNETES_NAMESPACE}
--conf spark.kubernetes.authenticate.driver.serviceAccountName=${KUBERNETES_SERVICE_ACCOUNT}
--conf spark.kubernetes.driver.container.image=liupengfei99/sparkstreamincv:latest
--conf spark.kubernetes.executor.container.image=liupengfei99/sparkstreamincv:latest
--conf spark.driver.cores=4
--conf spark.driver.memory=4096M
--conf spark.executor.cores=1
--conf spark.executor.memory=4096M
--conf spark.executor.instances=2
--conf spark.kubernetes.driver.podTemplateFile=/path/to/pod_template.yaml
--conf spark.kubernetes.executor.podTemplateFile=/path/to/pod_template.yaml
--conf spark.kubernetes.executor.deleteOnTermination=false
--class org.apache.spark.examples.SparkPi
local:///opt/spark/examples/jars/spark-examples_2.12-3.0.0-SNAPSHOT.jar 100
```

### Mode notebook
```python
spark = SparkSession \
    .builder.master("k8s://https://kubernetes.default.svc:443") \
    .appName("SparkStreamingComputerVision") \
    .config("spark.kubernetes.container.image", "liupengfei99/sparkstreamincv:latest") \
    .config("spark.kubernetes.authenticate.driver.serviceAccountName", os.environ['KUBERNETES_SERVICE_ACCOUNT']) \
    .config("spark.driver.cores","4")\
    .config("spark.driver.memory","4096M") \
    .config("spark.executor.cores","1") \
    .config("spark.executor.memory","4096M") \
    .config("spark.executor.instances", "2") \
    .config("spark.kubernetes.namespace", os.environ['KUBERNETES_NAMESPACE']) \
    .config("spark.kubernetes.executor.podTemplateFile","/path/to/pod_template.yaml") \
    .config("spark.kubernetes.driver.podTemplateFile","/path/to/pod_template.yaml") \
    .getOrCreate()

```

## Summery

With pod template, users will be more flexible when configuring Driver/Executor Pods, but Spark itself does not 
verify the correctness of PodTemplate, so this also brings a lot of trouble to debugging. Regarding NodeSelector, 
Taints, Tolerations, etc., it is more convenient to set these fields in Spark Operator.



