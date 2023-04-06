
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


def load_data():

    train_path = "/Users/macair/Downloads/FYP/faces/train"
    test_path = "/Users/macair/Downloads/FYP/faces/test"

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


with open('generators.pickle', 'wb') as f:
    pickle.dump((train_generator, test_generator, validation_generator), f)


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2*(precision*recall)/(precision+recall+K.epsilon())


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    f1_score,
]

ga = GA.GeneticAlgorithm(pop_size=50, num_generations=100,
                         mutation_rate=0.01, crossover_rate=0.8, gene_length=10)

ga.evolve_population()
best_individual = ga.best_genotype()


def define_model():
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
    model.compile(optimizers.Adam(learning_rate=0.0001,
                  decay=1e-4), loss='MSE', metrics=METRICS)
    model.summary()
    model.save('mymodel.h5')


def train_model():
    checkpoint = ModelCheckpoint("resnet50.h5", monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
    early = EarlyStopping(verbose=1, monitor='val_loss',
                          min_delta=0.0001, patience=20, mode='auto')
    lrd = ReduceLROnPlateau(monitor='val_loss', patience=20,
                            verbose=1, factor=0.50, min_lr=1e-10)
    prediction = model.predict(validation_generator)
    hist = model.fit(train_generator,
                     validation_data=validation_generator,
                     epochs=5,
                     callbacks=[checkpoint, early, lrd],
                     verbose=1)

# filename = "./Completed_model"
# save_model(model, filename)
# loaded_model = keras.models.load_model(filename)


def Train_Val_Plot(acc, val_acc, loss, val_loss, auc, val_auc, precision, val_precision, f1, val_f1):

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


Train_Val_Plot(hist.history['accuracy'], hist.history['val_accuracy'],
               hist.history['loss'], hist.history['val_loss'],
               hist.history['auc'], hist.history['val_auc'],
               hist.history['precision'], hist.history['val_precision'],
               hist.history['f1_score'], hist.history['val_f1_score']
               )

y_pred = model.predict(test_generator)
print(y_pred)

# %%


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# Plot normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=faces,
                      normalize=True, title='Normalized confusion matrix')
plt.show()
