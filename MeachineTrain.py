# 特征分类部分
from collections import Counter
import pandas as pd
import numpy as np
def dataprocessing(filepath):
    print("Loading feature files")
    dataset1 = pd.read_csv(filepath[0], header=None, low_memory=False)
    dataset2 = pd.read_csv(filepath[1], header=None, low_memory=False)
    dataset3 = pd.read_csv(filepath[2], header=None, low_memory=False)
    dataset4 = pd.read_csv(filepath[3], header=None, low_memory=False)
    # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset1=pd.DataFrame(dataset1,dtype=np.float)
    print("Feature processing")
    print(type(dataset1))
    # dataset1 = dataset1.convert_objects(convert_numeric=True)
    # dataset2 = dataset2.convert_objects(convert_numeric=True)
    # dataset3 = dataset3.convert_objects(convert_numeric=True)
    # dataset4 = dataset4.convert_objects(convert_numeric=True)
    dataset1 = dataset1.apply(pd.to_numeric, errors="ignore")
    dataset2 = dataset2.apply(pd.to_numeric, errors="ignore")
    dataset3 = dataset3.apply(pd.to_numeric, errors="ignore")
    dataset4 = dataset4.apply(pd.to_numeric, errors="ignore")

    dataset1.dropna(inplace=True)
    dataset2.dropna(inplace=True)
    dataset3.dropna(inplace=True)
    dataset4.dropna(inplace=True)
    ###filepath = [negtest_method, negtrain_method, postest_method, postrain_method]
    traindata = pd.concat([dataset2, dataset4], axis=0)
    testdata = pd.concat([dataset1, dataset3])

    # smo = SMOTE(random_state=42)

    negtraintags = [0] * dataset2.shape[0]
    postraintags = [1] * dataset4.shape[0]
    traintags = negtraintags + postraintags
    # print('the type of traintags is',type(traintags))
    # testdata = pd.concat([dataset1, dataset3])
    negtesttags = [0] * dataset1.shape[0]
    postesttags = [1] * dataset3.shape[0]
    testtags = negtesttags + postesttags
    # print('the type of testtags is',type(testtags))
    print('the before is', Counter(traintags))
    # traindata, traintags = smo.fit_sample(traindata, traintags)
    print('the after is ', Counter(traintags))
    testtags = np.array(testtags)
    print('the type of testtags is', type(testtags))
    traindata = np.array(traindata)
   # print("the shaoe of train data is ",traindata.shape)
    traintags = np.array(traintags)
   # print("the shaoe of train tag is ", traintags.shape)
    testdata = np.array(testdata)
    testtags = np.array(testtags)
    # traindata=traindata.reshape(traindata.shape[0],1,traindata[1])
    # testdata=testdata.reshape(testdata.shape[0],1,testdata[1])
    data = [traindata, traintags, testdata, testtags]
    #print("data shape is ",data)
    return data


# 得到路径下所有数据编号
def matchfiles(tmpdir, suffix):  # 读取文件路径
    ###windows os
    # f = glob.glob(tmpdir + '\\*.' + suffix)
    ###linux os
    fi = []
    filenames = []
    # f = glob.glob(tmpdir + suffix)
    f = glob.glob(tmpdir + '/*.' + suffix)
    return f


# 合并编号对应阳阴训练集或测试集并保存
def datadic(filegroup):
    # method = ["-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-RT.csv", "-PSSM-DT.csv", "-CC-PSSM.csv",
    # "-AC-PSSM.csv", "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
    # "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv"]
    method = ["-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-DT.csv", "-CC-PSSM.csv",
              "-AC-PSSM.csv", "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
              "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv"]

    # method = ["AC.csv","ACC.csv","CC.csv","DP.csv","DR.csv","kemr-.csv","PC-PseAAC-.csv","PC-PseAAC-General.csv",
    #         "-PDT.csv","SC-PseAAC.csv","SC-PseAAC-Gnereal.csv"]

    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    postest = filegroup["postest"]
    negtest = filegroup["negtest"]
    file_method = {}
    filepath = []
    for methodname in method:
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break
        # 匹配出methodname对应的文件
        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break
        # 匹配出methodname对应的文件
        for k in postest:
            if methodname in k:
                postest_method = k
                break
        # 匹配出methodname对应的文件
        for l in negtest:
            if methodname in l:
                negtest_method = l
                break
        # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        filepath = [negtest_method, negtrain_method, postest_method, postrain_method]
        # print (filepath)
        file_method[methodname] = dataprocessing(filepath)

    return file_method


# lightgbm搭建
import lightgbm as lgb


def lightgbmTrainStag_train(traindata, traintags, testdata, testtags, params,method):
    model_path = "pretrain_Models\\light_model\\"

    train_data = lgb.Dataset(traindata, label=traintags, silent=True)
    # validation_data=lgb.Dataset(testdata,label=testtags)
    clf = lgb.train(params, train_data)  # lgb训练
    clf.save_model(model_path+"lightgbm_model"+method+".txt")

    train_label_pro = clf.predict(traindata, predict_disable_shape_check=True, num_iteration=clf.best_iteration)
    train_label_01 = train_label_pro.copy()
    y_pred_pro = clf.predict(testdata, num_iteration=clf.best_iteration, predict_disable_shape_check=True)
    y_pred_01 = y_pred_pro.copy()
    # y_raw=clf.predict(testdata,raw_score=True,num_iteration=clf.best_iteration)##得到原始概率矩阵
    # print y_raw
    #    y_score = y_pred
    #    print y_score

    for i in range(len(train_label_01)):
        if train_label_01[i] > 0.5:
            train_label_01[i] = 1
        else:
            train_label_01[i] = 0
    for i in range(len(y_pred_01)):
        if y_pred_01[i] > 0.5:
            y_pred_01[i] = 1
        else:
            y_pred_01[i] = 0

    # print(y_pred)
    # return (train_label,y_pred, evaluateLight(testtags,y_pred,y_raw))#返回
    #print("the final is", train_label_pro.shape, train_label_01.shape)
    return (train_label_pro, y_pred_pro, train_label_01, y_pred_01)


def lightgbmTrainStag_infer(traindata, traintags, testdata, testtags, params,method):
    model_path = "pretrain_Models\\light_model\\"

    train_data = lgb.Dataset(traindata, label=traintags, silent=True)
    # validation_data=lgb.Dataset(testdata,label=testtags)
    clf=lgb.Booster(model_file=model_path+"lightgbm_model"+method+".txt")

    train_label_pro = clf.predict(traindata, predict_disable_shape_check=True, num_iteration=clf.best_iteration)
    train_label_01 = train_label_pro.copy()
    y_pred_pro = clf.predict(testdata, num_iteration=clf.best_iteration, predict_disable_shape_check=True)
    y_pred_01 = y_pred_pro.copy()
    # y_raw=clf.predict(testdata,raw_score=True,num_iteration=clf.best_iteration)##得到原始概率矩阵
    # print y_raw
    #    y_score = y_pred
    #    print y_score

    for i in range(len(train_label_01)):
        if train_label_01[i] > 0.5:
            train_label_01[i] = 1
        else:
            train_label_01[i] = 0
    for i in range(len(y_pred_01)):
        if y_pred_01[i] > 0.5:
            y_pred_01[i] = 1
        else:
            y_pred_01[i] = 0

    # print(y_pred)
    # return (train_label,y_pred, evaluateLight(testtags,y_pred,y_raw))#返回
    #print("the final is", train_label_pro.shape, train_label_01.shape)
    return (train_label_pro, y_pred_pro, train_label_01, y_pred_01)
# 机器学习分类预测部分

from lightgbm.sklearn import LGBMClassifier


def trainmodel_GBM_train(datadic):
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
        print(data[1].shape)
        #        model.fit(data[0], data[1])

        #        model=svm_best_parameters_cross_validation(data[0], data[1])
        cls = LGBMClassifier()
        params = cls.get_params()
        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        # (y_pred_train, y_pred_test, metric) = lightgbmTrainStag(data[0], data[1], data[2], data[3], params)
        (y_pred_train_pro, y_pred_test_pro, y_pred_train_01, y_pred_test_01) = lightgbmTrainStag_train(data[0], data[1],
                                                                                                 data[2], data[3],
                                                                                                 params,method=i)
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
    #print(train_feature_pro_vector)
    # print(test_feature_vector)

    data_train_pro = train_feature_pro_vector.values
    data_test_pro = test_feature_pro_vector.values
    data_train_01 = train_feature_01_vector.values
    data_test_01 = test_feature_01_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return (data_train_pro, data_test_pro, data_train_01, data_test_01)

def trainmodel_GBM_infer(datadic):
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
        print(data[1].shape)
        #        model.fit(data[0], data[1])

        #        model=svm_best_parameters_cross_validation(data[0], data[1])
        cls = LGBMClassifier()
        params = cls.get_params()
        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        # (y_pred_train, y_pred_test, metric) = lightgbmTrainStag(data[0], data[1], data[2], data[3], params)
        (y_pred_train_pro, y_pred_test_pro, y_pred_train_01, y_pred_test_01) = lightgbmTrainStag_infer(data[0], data[1],
                                                                                                 data[2], data[3],
                                                                                                 params,method=i)
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
    #print(train_feature_pro_vector)
    # print(test_feature_vector)

    data_train_pro = train_feature_pro_vector.values
    data_test_pro = test_feature_pro_vector.values
    data_train_01 = train_feature_01_vector.values
    data_test_01 = test_feature_01_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return (data_train_pro, data_test_pro, data_train_01, data_test_01)