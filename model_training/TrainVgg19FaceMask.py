import os

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras

# set train test data set path
train_dir = '/mnt/hgfs/Centos7_share_folder/Face_Mask_Dataset/Train'
test_dir = '/mnt/hgfs/Centos7_share_folder/Face_Mask_Dataset/Test'
val_dir = '/mnt/hgfs/Centos7_share_folder/Face_Mask_Dataset/Validation'

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.keras.autolog()

# it defines how many time the model will do a pass on the training data set, each pass, the model will be updated a little
# one pass means "one forward pass and one backward pass of all the training examples
epochs = 20
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")


def main():
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Data augmentation
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
        train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128, 128),
                                                            class_mode='categorical', batch_size=32)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        validate_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128),
                                                               class_mode='categorical',
                                                               batch_size=32)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128),
                                                           class_mode='categorical',
                                                           batch_size=32)

        # build model
        vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        for layer in vgg19.layers:
            layer.trainable = False

        model = Sequential()
        model.add(vgg19)
        model.add(Flatten())
        model.add(Dense(2, activation='sigmoid'))
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

        # train the model
        history = model.fit(train_generator,
                            steps_per_epoch=len(train_generator) // 32,
                            epochs=epochs, validation_data=validate_generator,
                            validation_steps=len(validate_generator) // 32)

        # evaluate the model
        model.evaluate(test_generator)

        # save the model
        mlflow.keras.log_model(model, "model", keras_module="keras")
        # model.save('masknet.h5')


if __name__ == "__main__":
    main()
