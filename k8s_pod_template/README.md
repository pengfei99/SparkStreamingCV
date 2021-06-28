# Spark on k8s use pod template 

Before spark 3.0, it's difficult to conifgure driver, executor running env by configuring pod specification. At present, the latest Release version 2.4.5 does not support custom Pod configuration through PodTemplateAfter Spark 3.0. The way it supports is actually relatively simple, that is, there can be a PodTemplate. A file to describe the metadata/spec fields of Driver/Executor, so of course you can add some fields required for scheduling in the template file.



