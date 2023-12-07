import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
import os
import itertools
import glob
import numpy as np
import cv2
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix


class ModelBuilder:
    def _read_dataset(self, target_dir):
        images = []
        labels = []
        for dir_path in glob.glob(self.path_in + '/' + target_dir + '/*'):
            label = dir_path.split('/')[-1]
            for img_path in glob.glob(os.path.join(dir_path, '*')):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

    def _preprocessing_label(self, train_labels, test_labels):
        self.le.fit(train_labels)
        train_labels_encoded = self.le.transform(train_labels)
        self.le.fit(test_labels)
        test_labels_encoded = self.le.transform(test_labels)
        return train_labels_encoded, test_labels_encoded

    def _split_dataset(self):
        [train_images, train_labels] = self._read_dataset("train")
        [test_images, test_labels] = self._read_dataset("test")
        [train_labels_encoded, test_labels_encoded] = self._preprocessing_label(train_labels=train_labels,
                                                                                test_labels=test_labels)
        X_train, Y_train, X_test, Y_test = train_images, train_labels_encoded, test_images, test_labels_encoded
        return X_train, Y_train, X_test, Y_test

    def __init__(self, path_dataset, model_name, load=False):
        self.IMAGE_SIZE = 224
        self.EPOCHS = 50
        self.n_PCA_components = 300
        self.pca = PCA(n_components=self.n_PCA_components)

        self.path_in = path_dataset
        self.loaded = load
        self.le = preprocessing.LabelEncoder()
        self.classNames = np.array(sorted(os.listdir(os.path.join(self.path_in, 'train'))))
        self.path_out = '{}/saved_model/{}'.format(os.getcwd(), model_name + '.h5')
        self.model_weight = '{}/saved_model/{}'.format(os.getcwd(), model_name + '_weight.h5')
        self.assets_dir = '{}/assets/'.format(os.getcwd())
        self.trained_model = tf.keras.applications.InceptionV3(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
                                                               include_top=False, weights='imagenet')
        self.model = Sequential() if not self.loaded else load_model(self.path_out)
        for layer in self.trained_model.layers:
            layer.trainable = False
        self.last_layer = self.trained_model.get_layer('mixed7')
        self.last_output = self.last_layer.output

        [self.X_train, self.Y_train, self.X_test, self.Y_test] = self._split_dataset()
        self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0
        self.history = None
        self.loss, self.acc = None, None

    def y_one_hot(self):
        return one_hot_encoder(self.Y_train), one_hot_encoder(self.Y_test)

    def features_extractor(self):
        train_feature_extractor = self.trained_model.predict(self.X_train)
        test_features_extractor = self.trained_model.predict(self.X_test)
        train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
        test_features = test_features_extractor.reshape(test_features_extractor.shape[0], -1)
        return train_features, test_features

    def pca_variance_test(self):
        """
        Choose the n_components for PCA
        """
        [train_features, _] = self.features_extractor()
        self.pca.fit(train_features)

        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.ylabel("Cum variance")
        # plt.show()
        pca_variance_image = os.path.join(self.assets_dir, 'PCA_variance.png')
        plt.savefig(pca_variance_image)

    def pca_fit_transform(self):
        [train_features, test_features] = self.features_extractor()

        train_PCA = self.pca.fit_transform(train_features)
        test_PCA = self.pca.fit_transform(test_features)
        return train_PCA, test_PCA

    def build_model(self):
        if not os.path.exists(self.path_out):
            inputs = Input(shape=self.n_PCA_components)
            hidden = Dense(256, activation='relu')(inputs)
            output = Dense(2, activation='sigmoid')(hidden)

            self.model = Model(inputs=inputs, outputs=output)
            self.model.compile(optimizer=RMSprop(learning_rate=0.0001),
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
        else:
            self.model = load_model(self.path_out)
            if os.path.exists(self.model_weight):
                self.model.load_weights(self.model_weight)
        self.model.summary()

    def train_model(self):
        [train_images, _] = self._read_dataset("train")
        [train_PCA, _] = self.pca_fit_transform()
        [y_train_one_hot, _] = self.y_one_hot()
        self.history = self.model.fit(train_PCA,
                                      y_train_one_hot,
                                      steps_per_epoch=len(train_images),
                                      epochs=self.EPOCHS,
                                      verbose=1)

    def evaluate_model(self):
        [_, test_PCA] = self.pca_fit_transform()
        [_, y_test_one_hot] = self.y_one_hot()
        self.loss, self.acc = self.model.evaluate(test_PCA, y_test_one_hot)

        plt.plot(self.history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train'], loc='upper left')
        # plt.show()
        loss_image = os.path.join(self.assets_dir, 'Model_Loss.png')
        plt.savefig(loss_image)
        plt.clf()

        plt.plot(self.history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train'], loc='upper left')
        # plt.show()
        accuracy_image = os.path.join(self.assets_dir, 'Model_Accuracy.png')
        plt.savefig(accuracy_image)
        plt.clf()

    def save_model(self):
        if not self.loaded:
            self.model.save(self.path_out)
        self.model.save_weights(self.model_weight)

    def confusion_matrix_pred(self, target_PCA):
        pred = self.model(target_PCA)
        pred = np.argmax(pred, axis=1)
        pred = self.le.inverse_transform(pred)

        predict = []
        for i in range(len(pred)):
            if pred[i] == 'fire_images':
                predict.append(1)
            else:
                predict.append(0)
        return predict

    def model_confusion(self):
        [train_PCA, test_PCA] = self.pca_fit_transform()
        train_predict = self.confusion_matrix_pred(train_PCA)
        test_predict = self.confusion_matrix_pred(test_PCA)
        [_, train_labels] = self._read_dataset("train")
        [_, test_labels] = self._read_dataset("test")
        [train_labels_encoded, test_labels_encoded] = self._preprocessing_label(train_labels=train_labels,
                                                                                test_labels=test_labels)
        train_cm = confusion_matrix(train_labels_encoded, train_predict)
        plot_confusion_matrix(cm=train_cm, target="train", assets_dir=self.assets_dir, classes=self.classNames,
                              title='Confusion Matrix Training')
        plt.close()
        test_cm = confusion_matrix(test_labels_encoded, test_predict)
        plot_confusion_matrix(cm=test_cm, target="test", assets_dir=self.assets_dir, classes=self.classNames,
                              title='Confusion Matrix Testing')
        plt.close()


def one_hot_encoder(target):
    return to_categorical(target)


def plot_confusion_matrix(cm, assets_dir, target, classes, normalize=False, title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    confusion_image = os.path.join(assets_dir, 'Confusion_Matrix_{}.png'.format(target))
    plt.savefig(confusion_image)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
