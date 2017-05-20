import txt_utils
import image_utils
import os
from keras import layers
from keras.models import Sequential
from keras import backend as K
import numpy
import sys

index = 0
def data_generator(image_dir_path, txt_dir_path):
    #global index
    image_paths = os.listdir(image_dir_path)
    txt_paths = os.listdir(txt_dir_path)
    for image_path in image_paths:
        try:
            image_vector = numpy.array([image_utils.image_to_numpy(image_dir_path+"/"+image_path)])
            if numpy.random.random() > 0.5:
                label = 1.0
                txt_file_name = txt_dir_path + "/" + image_path.split(".")[0] +".txt"
                txt_vector = txt_utils.query("d2vModel", txt_file_name)
            else:
                label = -1.0
                txt_file_name = txt_dir_path + "/" + numpy.random.choice(txt_paths)
                txt_vector = txt_utils.query("d2vModel", txt_file_name)
            txt_vector = numpy.reshape(txt_vector, (1, 256))
            '''
            stdout = sys.stdout
            f = open("indexs", "a")
            sys.stdout = f
            print(index)
            sys.stdout = stdout
            index += 1
            '''
            yield [image_vector, txt_vector], [numpy.array([label])]
        except:
            pass


def loss(y_true, y_predict):
    return K.mean(K.maximum(K.square(y_predict)*y_true+1, 0))

def build_cnn_dnn():
    # cnn model
    cnn_model = Sequential()
    cnn_model.add(layers.convolutional.Convolution2D(filters=128, kernel_size=(3, 3), padding="valid",
                                   #                  kernel_initializer='glorot_uniform',
                                                     activation="relu", input_shape=(256, 256, 3)))
    cnn_model.add(layers.convolutional.Conv2D(filters=64, kernel_size=(3, 3), padding="valid",
                                   #           kernel_initializer='glorot_uniform',
                                              activation="relu"))
    cnn_model.add(layers.pooling.MaxPool2D(pool_size=(5, 5)))
    cnn_model.add(layers.convolutional.Conv2D(filters=64, kernel_size=(3, 3), padding="valid",
                                   #           kernel_initializer='glorot_uniform',
                                              activation="relu"))
    cnn_model.add(layers.pooling.MaxPool2D(pool_size=(4, 4)))
    cnn_model.add(layers.convolutional.Conv2D(filters=64, kernel_size=(3, 3), padding="valid",
                                   #           kernel_initializer='glorot_uniform',
                                              activation="relu"))
    cnn_model.add(layers.pooling.MaxPool2D(pool_size=(5, 4)))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(128))
    cnn_model.add(layers.Activation("relu"))

    # dnn_model
    dnn_model = Sequential()
    dnn_model.add(layers.Dense(256, input_shape=(256, ), activation='relu'))
    dnn_model.add(layers.Dense(256, activation='relu'))
    dnn_model.add(layers.Dense(128, activation='relu'))
    # merge
    total_model = Sequential()
    total_model.add(layers.Merge([cnn_model, dnn_model], mode="sum"))
    total_model.compile(loss=loss, optimizer="adam")
    return total_model

def fit_cnn_dnn(model, image_dir_path, txt_dir_path):
    model.fit_generator(data_generator(image_dir_path, txt_dir_path),
                        samples_per_epoch=200,
                        epochs=400)
    model.save("cnn_dnn_model")
    return model




if __name__ == "__main__":
    txt_dir_path = "/media/rw/DATA/sogou/formalCompetition4/News_info_train"
    image_dir_path = "/media/rw/DATA/sogou/formalCompetition4/News_pic_info_train"
    fit_cnn_dnn(build_cnn_dnn(), image_dir_path, txt_dir_path)
