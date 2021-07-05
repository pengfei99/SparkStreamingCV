import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Step1: Ingest, normalize, and augment data
## Download data
_URL = 'https://minio.lab.sspcloud.fr/pengfei/diffusion/computer_vision/model_training_data/Face_Mask_Dataset.zip'
path_to_zip = tf.keras.utils.get_file('Face_Mask_Dataset.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'Face_Mask_Dataset')

train_dir = os.path.join(PATH, 'Train')
validation_dir = os.path.join(PATH, 'Validation')
test_dir = os.path.join(PATH, 'Test')

# batch_size: Size of the batches of data. Default: 32.
# image_size: Size to resize images to after they are read from disk.
#         Defaults to `(256, 256)`.
BATCH_SIZE = 64
IMG_SIZE = (128, 128)

# it defines how many time the model will do a pass on the training data set, each pass, the model will be
# updated a little one pass means "one forward pass and one backward pass of all the training examples

## normalize training data
# This function will organize training images in batch of 64, each image will be normalized to size
# (128,128). with_mask and with_no_mask images will be randomly shuffled.
train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

# show an example of the training dataset
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

## normalize validation and test data
validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory(test_dir,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)

## Dataset performance tuning
# Tune the performance of training dataset by using buffered prefetching to load images from disk
# to avoid disk I/O blockage.

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

## Data augumentation
# When you don't have a large image dataset, it's a good practice to artificially introduce
# sample diversity by applying random, yet realistic, transformations to the training images,
# such as rotation and horizontal flipping. This helps expose the model to different aspects
# of the training data and reduce overfitting.

# RandomFlip : define how to rotate image randomly
# RandomRotation:           (Shear angle in counter-clockwise direction in degrees)
# RandomZoom: It takes a Float or [lower, upper]. It defines the Range for random zoom.
#           If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
    plt.show()

## Rescale image pixel value
# The tensorflow.keras.applications.vgg19 base model expects pixel values in [0, 1], but at this point, the
# pixel values in our images are in [0, 255]. To rescale them, we need a rescale layer to rescale image pixel
# from [0, 255] to [0, 1]


rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
# Note some models expect pixel values in [-1,1], then your rescale layer should looks like this
# rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

# Summary, all above steps are used to ingest, normalize and augment data. To your code cleaner, we can use
# train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True,
#                                                                     zoom_range=0.2, shear_range=0.2)
# train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128, 128),
#                                                         class_mode='categorical', batch_size=batch)
#
# validation_data_gene = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
# validate_generator = val_datagen.flow_from_directory(directory=validation_dir, target_size=(128, 128),
#                                                          class_mode='categorical',
#                                                          batch_size=32)
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
# test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(128, 128),
#                                                       class_mode='categorical',
#                                                       batch_size=32)

############################# Step2: Build and Train our model ########################################################

# In this chapter, we will use a vgg19 model from keras. You can find the Reference paper:
# [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)

# The weights of the model is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and
# 1000 classes. ImageNet is a research training dataset with a wide variety of categories.

# First, you need to pick which layer of Vgg19 you will use for feature extraction. The very last classification
# layer (on "top", as most diagrams of machine learning models go from bottom to top) is not very useful. Instead,
# you will follow the common practice to depend on the very last layer before the flatten operation. This layer
# is called the "bottleneck layer". The bottleneck layer features retain more generality as compared to the
# final/top layer.

# We will follow the below step:
# 1. Instantiate a Vgg19 model and pre-loaded it with weights trained on ImageNet. By specifying the include_top=False
#    argument, we load a network that doesn't include the classification layers at the top, which is ideal for
#    feature extraction.

# For image shape, we need to add color channel to image size
IMG_SHAPE = IMG_SIZE + (3,)
base_model = VGG19(input_shape=IMG_SHAPE,
                   include_top=False,
                   weights='imagenet')

# 2. Build our model by using the base_model
# freeze base_model layers
for layer in base_model.layers:
    layer.trainable = False

# create our model
model = Sequential()
# add rescaling layer
model.add(rescale)

# add base model to our model as button layers
model.add(base_model)
# add a flatten layer to flat the output of button layers
model.add(Flatten())

# add a classification layer as output layer
model.add(Dense(2, activation='sigmoid'))

# compile your model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

# view the model summary after compile/build
model.summary()
# train the model
epochs = 2
history = model.fit(train_dataset,
                    steps_per_epoch=len(train_dataset) // 32,
                    epochs=epochs, validation_data=validation_dataset,
                    validation_steps=len(validation_dataset) // 32)

# evaluate the model
model.evaluate(test_dataset)

# save the model
# model.save('masknet.h5')


#
# def main():
#     # Data augmentation

#
#     # build model
#     vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
#     for layer in vgg19.layers:
#         layer.trainable = False
#
#     model = Sequential()
#     model.add(vgg19)
#     model.add(Flatten())
#     model.add(Dense(2, activation='sigmoid'))
#     model.summary()
#     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
#
#     # train the model
#     history = model.fit(train_generator,
#                         steps_per_epoch=len(train_generator) // 32,
#                         epochs=epochs, validation_data=validate_generator,
#                         validation_steps=len(validate_generator) // 32)
#
#     # evaluate the model
#     model.evaluate(test_generator)
#
#     # save the model
#     # model.save('masknet.h5')


# if __name__ == "__main__":
#     main()
