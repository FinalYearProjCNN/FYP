import GA
import pickle
import nbimporter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
import seaborn as sn
from sklearn.model_selection import train_test_split
import skimage.io
from sklearn import metrics
import keras.backend as K
import keras
import keras.layers as layers
from keras import Model, Sequential
import keras.models
from keras import optimizers, applications
from keras.applications.nasnet import NASNetLarge
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from GA import GeneticAlgorithm


class MyModel():

    def load_data(self):
        classes = ['angry', 'disgusted', 'fearful',
                   'happy', 'neutral', 'sad', 'suprised']
        train_path = "/Users/macair/FYP_TitoE/FYP/faces/train"
        test_path = "/Users/macair/FYP_TitoE/FYP/faces/test"

        batch_size = 64
        train_datagen = image.ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=0.2
        )

        test_datagen = image.ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            width_shift_range=0.1,
            height_shift_range=0.1)

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(48, 48),
            batch_size=64,
            class_mode='categorical',
            shuffle=True)

        validation_generator = train_datagen.flow_from_directory(
            train_path,
            subset='validation',
            target_size=(48, 48),
            batch_size=64,
            shuffle=True,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(48, 48),
            batch_size=64,
            shuffle=True,
            class_mode='categorical')

    def f1_score(self, y_pred):
        true_positives = K.sum(K.round(K.clip(self * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(self, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        return 2*(precision*recall)/(precision+recall+K.epsilon())

    ga = GA.GeneticAlgorithm(pop_size=50, num_generations=100,
                             mutation_rate=0.01, crossover_rate=0.8, gene_length=10)

    ga.evolve_population()

    def __init__(self):
        base_model = applications.ResNet50(weights='imagenet',
                                           include_top=False,
                                           input_shape=(48, 48, 3))

        for layer in base_model.layers[:-4]:
            layer.trainable = False

            model = keras.Sequential()
            model.add(base_model)
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1024, activation="relu"))
            model.add(keras.layers.Dense(7, activation="softmax"))
            model.add(keras.layers.Dropout(0.5))

        input_shape = (None, 48, 48, 3)
        model.build(input_shape)

        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            self.f1_score,
        ]

        model.compile(optimizers.Adam(learning_rate=0.0001,
                                      decay=1e-4), loss='MSE', metrics=METRICS)
        model.summary()
        model.save('/Users/macair/FYP_TitoE/FYP/Completed_model/mymodel.h5')

    def train_model(self):
        checkpoint = ModelCheckpoint("resnet50.h5", monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
        early = EarlyStopping(verbose=1, monitor='val_loss',
                              min_delta=0.0001, patience=20, mode='auto')
        lrd = ReduceLROnPlateau(
            monitor='val_loss', patience=20, verbose=1, factor=0.50, min_lr=1e-10)
        prediction = self.model.predict(self.validation_generator)
        return self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=5,
            callbacks=[checkpoint, early, lrd],
            verbose=1,
        )

    # filename = "./Completed_model"
    # save_model(model, filename)
    # loaded_model = keras.models.load_model(filename)

    def Train_Val_Plot(self, acc, val_acc, loss, val_loss, auc, val_auc, precision, val_precision, f1, val_f1):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

        ax1.plot(range(1, len(acc) + 1), acc)
        ax1.plot(range(1, len(val_acc) + 1), val_acc)
        ax1.set_title('History of Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend(['training', 'validation'])

        ax2.plot(range(1, len(loss) + 1), loss)
        ax2.plot(range(1, len(val_loss) + 1), val_loss)
        ax2.set_title('History of Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend(['training', 'validation'])

        ax3.plot(range(1, len(auc) + 1), auc)
        ax3.plot(range(1, len(val_auc) + 1), val_auc)
        ax3.set_title('History of AUC')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('AUC')
        ax3.legend(['training', 'validation'])

        ax4.plot(range(1, len(precision) + 1), precision)
        ax4.plot(range(1, len(val_precision) + 1), val_precision)
        ax4.set_title('History of Precision')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Precision')
        ax4.legend(['training', 'validation'])

        ax5.plot(range(1, len(f1) + 1), f1)
        ax5.plot(range(1, len(val_f1) + 1), val_f1)
        ax5.set_title('History of F1-score')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('F1 score')
        ax5.legend(['training', 'validation'])

        plt.show()

    def train_and_plot(self):
        hist = self
