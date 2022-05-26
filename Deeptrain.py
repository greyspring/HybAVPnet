
from keras.layers import Conv1D, Dropout, Bidirectional, LSTM, Flatten, Multiply, Embedding, AveragePooling1D, \
    MaxPooling1D, GlobalAveragePooling1D
from keras.layers.core import *
import keras.backend as K
from keras.models import *
from keras import optimizers
from utils import *
import pandas as pd
import numpy as np
def buildmodel(traindata):
    model = Sequential()
    # model.add(Embedding(2,1,input_length=traindata[1].shape))
    # model.add(Self_Attention(128))
    timesteps = traindata.shape[1]
    dims = traindata.shape[2]
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu', strides=1, input_shape=(timesteps, dims)))
    # model.add(AveragePooling1D(pool_size =2))
    # model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    # model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def deeptags_train(traindata, traintags, testdata, testtags,method):
    model_path = "pretrain_Models\\Deep_model\\"
    model = buildmodel(traindata)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
    history = LossHistory()
    model.fit(traindata, traintags, batch_size=35, epochs=50, callbacks=[history])
    model.save(model_path+"deep_model"+method+".h5")
    train_label_pro = model.predict(traindata)
    y_pred_pro = model.predict(testdata)
    train_label_01 = model.predict_classes(traindata)
    y_pred_01 = model.predict_classes(testdata)
    print('the shape is ', train_label_pro.shape, train_label_01.shape)
    return (train_label_pro.flatten(), y_pred_pro.flatten(), train_label_01.flatten(), y_pred_01.flatten())

def deeptags_infer(traindata, traintags, testdata, testtags,method):
    model_path = "pretrain_Models\\Deep_model\\"
    model=load_model(model_path+"deep_model"+method+".h5")
    train_label_pro = model.predict(traindata)
    y_pred_pro = model.predict(testdata)
    train_label_01 = model.predict_classes(traindata)
    y_pred_01 = model.predict_classes(testdata)
    print('the shape is ', train_label_pro.shape, train_label_01.shape)
    return (train_label_pro.flatten(), y_pred_pro.flatten(), train_label_01.flatten(), y_pred_01.flatten())

def trainmodel_deep_train(datadic):
    train_feature_pro = {}
    test_feature_pro = {}
    train_feature_01 = {}
    test_feature_01 = {}
    for i in datadic:  # 每一轮代表每一种特征提取方法
        data = datadic[i]
        # index.append(i)
        #        model = svm.SVC(probability=False)
        #        print("Svm training")

        print(i, '方法所的该数据集的规模为：')
        print(data[0].shape)
        print(data[2].shape)
        #        model.fit(data[0], data[1])

        #        model=svm_best_parameters_cross_validation(data[0], data[1])

        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        # (y_pred_train, y_pred_test, metric) = lightgbmTrainStag(data[0], data[1], data[2], data[3], params)
        data[0] = data[0].reshape(data[0].shape[0], 1, data[0].shape[1])
        data[2] = data[2].reshape(data[2].shape[0], 1, data[2].shape[1])
        (y_pred_train_pro, y_pred_test_pro, y_pred_train_01, y_pred_test_01) = deeptags_train(data[0], data[1], data[2],
                                                                                        data[3],method=i)
        train_feature_pro[i] = y_pred_train_pro
        test_feature_pro[i] = y_pred_test_pro
        train_feature_01[i] = y_pred_train_01
        test_feature_01[i] = y_pred_test_01
        print("训练结果所得标签规模为")
        print(train_feature_pro[i].shape)
        print(test_feature_pro[i].shape)
        # data = [traindata,traintags,testdata,testtags]
        print('#################################################')

    train_feature_pro_vector = pd.DataFrame(train_feature_pro)
    test_feature_pro_vector = pd.DataFrame(test_feature_pro)
    train_feature_01_vector = pd.DataFrame(train_feature_01)
    test_feature_01_vector = pd.DataFrame(test_feature_01)
    # print('the metric of the trainfeatr vector and test is ')
    print(train_feature_pro_vector)
    # print(test_feature_vector)

    data_train_pro = train_feature_pro_vector.values
    data_test_pro = test_feature_pro_vector.values
    data_train_01 = train_feature_01_vector.values
    data_test_01 = test_feature_01_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return (data_train_pro, data_test_pro, data_train_01, data_test_01)

def trainmodel_deep_infer(datadic):
    train_feature_pro = {}
    test_feature_pro = {}
    train_feature_01 = {}
    test_feature_01 = {}
    for i in datadic:  # 每一轮代表每一种特征提取方法
        data = datadic[i]
        # index.append(i)
        #        model = svm.SVC(probability=False)
        #        print("Svm training")

        print(i, '方法所的该数据集的规模为：')
        print(data[0].shape)
        print(data[2].shape)
        #        model.fit(data[0], data[1])

        #        model=svm_best_parameters_cross_validation(data[0], data[1])

        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        # (y_pred_train, y_pred_test, metric) = lightgbmTrainStag(data[0], data[1], data[2], data[3], params)
        data[0] = data[0].reshape(data[0].shape[0], 1, data[0].shape[1])
        data[2] = data[2].reshape(data[2].shape[0], 1, data[2].shape[1])
        (y_pred_train_pro, y_pred_test_pro, y_pred_train_01, y_pred_test_01) = deeptags_infer(data[0], data[1], data[2],
                                                                                        data[3],method=i)
        train_feature_pro[i] = y_pred_train_pro
        test_feature_pro[i] = y_pred_test_pro
        train_feature_01[i] = y_pred_train_01
        test_feature_01[i] = y_pred_test_01
        print("训练结果所得标签规模为")
        print(train_feature_pro[i].shape)
        print(test_feature_pro[i].shape)
        # data = [traindata,traintags,testdata,testtags]
        print('#################################################')

    train_feature_pro_vector = pd.DataFrame(train_feature_pro)
    test_feature_pro_vector = pd.DataFrame(test_feature_pro)
    train_feature_01_vector = pd.DataFrame(train_feature_01)
    test_feature_01_vector = pd.DataFrame(test_feature_01)
    # print('the metric of the trainfeatr vector and test is ')
    print(train_feature_pro_vector)
    # print(test_feature_vector)

    data_train_pro = train_feature_pro_vector.values
    data_test_pro = test_feature_pro_vector.values
    data_train_01 = train_feature_01_vector.values
    data_test_01 = test_feature_01_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return (data_train_pro, data_test_pro, data_train_01, data_test_01)