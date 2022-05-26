#onehot
from OnehotTrain import Gentestdata,Gentraindata,buildmodel2
from utils import LossHistory

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
from MeachineTrain import dataprocessing,datadic
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
datadics = datadic(filegroup)#返回所有特征提取方法的数据集filemethod{methodname:data=[triandata,traintags,testdata,testtags]}
print(datadics)
#print(data[0].shape,data[2].shape)

#meachien
#meachien

from OnehotTrain import Gentestdata, Gentraindata, buildmodel2, load_model
from MeachineTrain import trainmodel_GBM_train
from Deeptrain import trainmodel_deep_train
import numpy as np
model.fit(traindataMat, trainlabelArr, batch_size=50, epochs=20, callbacks=[history])
model_path = "pretrain_Models\\One-hot_model\\"
model.save(model_path+"deep_model"+"one-hot"+".h5")
(train1,test1,train2,test2)=trainmodel_GBM_train(datadics)
(train3,test3,train4,test4)=trainmodel_deep_train(datadics)

#model_path = "pretrain_Models\\One-hot_model\\"
#model=load_model(model_path+"deep_model"+"one-hot"+".h5")

Exdata1=model.predict(traindataMat)
Exdata2=model.predict(testdataMat)
Ec1=model.predict_classes(traindataMat)
Ec2=model.predict_classes(testdataMat)
Exdata1=model.predict(traindataMat)
Exdata2=model.predict(testdataMat)
Ec1=model.predict_classes(traindataMat)
Ec2=model.predict_classes(testdataMat)

tmp1=np.concatenate((train1,train2,train3,train4,Exdata1,Ec1),axis=1)
tmp2=np.concatenate((test1,test2,test3,test4,Exdata2,Ec2),axis=1)

