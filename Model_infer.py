#onehot
import joblib

from Deeptrain import trainmodel_deep_infer
from OnehotTrain import Gentestdata, Gentraindata, buildmodel2, load_model
from evaluate import find_best_SVM
from utils import LossHistory
import numpy as np
dataTrainLabel = [0, 1]
dataTrainFilePaths = ['AVP_data_2\AnBP2-NonAVP544-train.fasta',  'AVP_data_2\Training_AVP544p.fasta']

traindataMat,trainlabelArr=Gentraindata(dataTrainFilePaths,dataTrainLabel,50)

dataTestFilePaths = ['AVP_data_2\AnBP2-NonAVP60-valid.fasta','AVP_data_2\\valid_AVP60.fasta']

dataTestLabel = [0, 1]
testdataMat,testlabelArr=Gentestdata(dataTestFilePaths,dataTestLabel,50)
print(traindataMat.shape)
###
import os
import tensorflow as tf


##
model = buildmodel2()
print(model.summary())
history = LossHistory()

#data dicting
from MeachineTrain import dataprocessing, datadic, trainmodel_GBM_infer
import glob
#tmpdir  ="method-feature-1"
tmpdir  ="method-feature-2"

#postrain = glob.glob(tmpdir + '/Training_AVP544p*')
#negtrain = glob.glob(tmpdir + '/Train_NonAVP407*')
#postest = glob.glob(tmpdir + '/valid_AVP60*')
#negtest = glob.glob(tmpdir + '/valid_nonAVP45*')

postrain = glob.glob(tmpdir + '/Training_AVP544p*')
negtrain = glob.glob(tmpdir + '/AnBP2-NonAVP544-train*')
postest = glob.glob(tmpdir + '/valid_AVP60*')
negtest = glob.glob(tmpdir + '/AnBP2-NonAVP60-valid*')
filegroup = {}
filegroup['postrain'] = postrain
filegroup['negtrain'] = negtrain
filegroup['postest'] = postest
filegroup['negtest'] = negtest
print(filegroup)
datadics = datadic(filegroup)#filemethod{methodname:data=[triandata,traintags,testdata,testtags]}
print(datadics)

model_path = "pretrain_Models\\One-hot_model\\"
model=load_model(model_path+"deep_model"+"one-hot"+".h5")
Exdata1=model.predict(traindataMat)
Exdata2=model.predict(testdataMat)
Ec1=model.predict_classes(traindataMat)
Ec2=model.predict_classes(testdataMat)

(train1,test1,train2,test2)=trainmodel_GBM_infer(datadics)
(train3,test3,train4,test4)=trainmodel_deep_infer(datadics)

tmp1=np.concatenate((train1,train2,train3,train4,Exdata1,Ec1),axis=1)
tmp2=np.concatenate((test1,test2,test3,test4,Exdata2,Ec2),axis=1)
#tmp1=np.concatenate((train1,train2),axis=1)
#tmp2=np.concatenate((test1,test2),axis=1)



from evaluate import SVM_plot, Rd_plot, knn_plot, SVM, Rd, knn
import pandas as pd
#dic1=SVM_plot(tmp1,trainlabelArr,tmp2,testlabelArr,filename='SVM')
#dic2=Rd_plot(tmp1,trainlabelArr,tmp2,testlabelArr,filename='Rd')
#dic3=knn_plot(tmp1,trainlabelArr,tmp2,testlabelArr,filename='knn')

dic1=SVM(tmp2,testlabelArr)
dic2=Rd(tmp1,trainlabelArr,tmp2,testlabelArr)
dic3=knn(tmp1,trainlabelArr,tmp2,testlabelArr)
metric1 = pd.DataFrame(dic1, index=(0, 5))
metric2 = pd.DataFrame(dic2, index=(0, 5))
metric3 = pd.DataFrame(dic3, index=(0, 5))

col = ['acc', 'sen', 'spec', 'mcc', 'f1_score']
piece1 = metric1.loc[0, col]
piece2 = metric2.loc[0, col]
piece3 = metric3.loc[0, col]
piece1.name="svm"
piece2.name='randomforest'
piece3.name='knn'
outcome=pd.concat([piece1,piece2,piece3],axis=1)
filename = "Independent_compare_SVM_RD_KNN_60_60times.csv"
filepath = "results\\" + filename
outcome.to_csv(filepath)

