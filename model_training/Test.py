import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

# download data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# this download data to ~/.keras, as we set extract to true, zip file are extracted automatically
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

print("Downloaded zip data path".format(path_to_zip))
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
print("Unzipped file path".format(PATH))

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

class_names = train_dataset.class_names

# show image in figure
# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#     plt.show()

# split validation data to test data
val_batches = tf.data.experimental.cardinality(validation_dataset)
print("original validation data set size: {}".format(val_batches))
# val_batches // 5 returns tf.Tensor(6, shape=(), dtype=int64)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('After split, the number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('After split, the number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

################## Configure the dataset for performance ##################################
# Use buffered prefetching to load images from disk without having I/O become blocking. To learn more about this
# method see the https://www.tensorflow.org/guide/data_performance.

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#### Data augumentation #################

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# train_dataset.take(1) returns <TakeDataset shapes: ((None, 160, 160, 3), (None,)), types: (tf.float32, tf.int32)>
# so the var image take the shapes, and var _ take the types
# show the added rotation layer of image
# for image, _ in train_dataset.take(1):
#     plt.figure(figsize=(10, 10))
#     first_image = image[0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#         plt.imshow(augmented_image[0] / 255)
#         plt.axis('off')
#     plt.show()

############################### Rescale pixel values ##################################
# Most of the model expect the pixels of the input image are in [-1, 1] or [0, 1]. But image pixels are often between
# in [0,255] for 8 bits. As a result we need to rescale the image for the moddel

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

############################ Create the base model from the pre-trained convnets #######################

"""
You will create the base model from the MobileNet V2 model developed at Google. This is pre-trained on the 
ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. ImageNet is a research training dataset 
with a wide variety of categories like jackfruit and syringe. This base of knowledge will help us classify cats 
and dogs from our specific dataset.

First, you need to pick which layer of MobileNet V2 you will use for feature extraction. The very last classification 
layer (on "top", as most diagrams of machine learning models go from bottom to top) is not very useful. Instead, 
you will follow the common practice to depend on the very last layer before the flatten operation. This layer 
is called the "bottleneck layer". The bottleneck layer features retain more generality as compared to the 
final/top layer.

First, instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet. By specifying the 
include_top=False argument, you load a network that doesn't include the classification layers at the top, 
which is ideal for feature extraction.
"""

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

""" 
This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features. Let's see what it does 
to an example batch of images:
"""

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

########################################## Feature extraction #####################################################
""" In this step, you will freeze the convolutional base created from the previous step and to use as a 
feature extractor. Additionally, you add a classifier on top of it and train the top-level classifier.

Freeze the convolutional base
It is important to freeze the convolutional base before you compile and train the model. Freezing 
(by setting layer.trainable = False) prevents the weights in a given layer from being updated during training. 
MobileNet V2 has many layers, so setting the entire model's trainable flag to False will freeze all of them.
"""
