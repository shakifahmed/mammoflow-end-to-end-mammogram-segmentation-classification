import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet50V2
from keras.applications.vgg16 import VGG16
from keras.applications.efficientnet import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, Flatten

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

class MyModel:
    @staticmethod
    def model():
        tf.random.set_seed(75)
        # base = ResNet50V2(include_top=False, weights=None, input_shape=(224,224,1))
        base =  VGG16(include_top=False, weights=None, input_shape=(224,224,1))
        base.trainable = True
        x = base.output
        x = Flatten()(x)
        outputs = Dense(units=8, activation='softmax')(x)

        model = Model(inputs=base.input, outputs=outputs)  
        model.summary()
        
        return model
    @staticmethod
    def compile(model, X_train_scaled_dl, y_train, X_test_scaled_dl, y_test):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
        history = model.fit(X_train_scaled_dl, y_train, epochs=10, batch_size=64, validation_data=(X_test_scaled_dl, y_test))
        return model, history
    @staticmethod
    def performance(model, X_test_scaled_dl, y_test):
        y_pred = model.predict(X_test_scaled_dl)
        y_pred_classes = np.argmax(y_pred, axis=1) 
        cm = confusion_matrix(y_test, y_pred_classes)
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='macro')
        recall = recall_score(y_test, y_pred_classes, average='macro')
        f1 = f1_score(y_test, y_pred_classes, average='macro')

        return cm, accuracy, precision, recall, f1
