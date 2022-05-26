from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef,roc_curve
from sklearn import svm
from sklearn.svm import SVC

def SVM_plot(c,gamma,traindata,trainlabel,testdata,testlabel,filename):
    model = SVC(C=c, gamma=gamma, probability=True)
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]

    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic={}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen']=recall_score(testlabel, prediction)
    dic['f1_score']=f1_score(testlabel, prediction)
    dic['spec']=spec
    dic['mcc']=matthews_corrcoef(testlabel, prediction)
    plotPR(testlabel,y_score[:,1],filename)
    plotROC(testlabel,y_score[:,1],filename)
    return dic
import joblib
def SVM(testdata,testlabel):
    #model = SVC(C=c, gamma=gamma, probability=True)
    #model.fit(traindata,trainlabel)
    model = joblib.load("my_model.m")
    prediction=model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]

    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic={}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen']=recall_score(testlabel, prediction)
    dic['f1_score']=f1_score(testlabel, prediction)
    dic['spec']=spec
    dic['mcc']=matthews_corrcoef(testlabel, prediction)
    return dic
from sklearn.ensemble import RandomForestClassifier

def Rd_plot(traindata,trainlabel,testdata,testlabel,filename):
    model= RandomForestClassifier(oob_score=True)
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]

    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic={}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen']=recall_score(testlabel, prediction)
    dic['f1_score']=f1_score(testlabel, prediction)
    dic['spec']=spec
    dic['mcc']=matthews_corrcoef(testlabel, prediction)
    plotPR(testlabel,y_score[:,1],filename)
    plotROC(testlabel,y_score[:,1],filename)
    return dic

def Rd(traindata,trainlabel,testdata,testlabel):
    model= RandomForestClassifier(oob_score=True)
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]

    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic={}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen']=recall_score(testlabel, prediction)
    dic['f1_score']=f1_score(testlabel, prediction)
    dic['spec']=spec
    dic['mcc']=matthews_corrcoef(testlabel, prediction)
    return dic
from sklearn.neighbors import  KNeighborsClassifier

def knn_plot(traindata,trainlabel,testdata,testlabel,filename):
    model = KNeighborsClassifier()
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]

    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic={}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen']=recall_score(testlabel, prediction)
    dic['f1_score']=f1_score(testlabel, prediction)
    dic['spec']=spec
    dic['mcc']=matthews_corrcoef(testlabel, prediction)
    plotPR(testlabel,y_score[:,1],filename)
    plotROC(testlabel,y_score[:,1],filename)
    return dic

def knn(traindata,trainlabel,testdata,testlabel):
    model = KNeighborsClassifier()
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]

    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic={}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen']=recall_score(testlabel, prediction)
    dic['f1_score']=f1_score(testlabel, prediction)
    dic['spec']=spec
    dic['mcc']=matthews_corrcoef(testlabel, prediction)
    return dic

import lightgbm as lgb
def lightgbm(traindata,trainlabel,testdata,testlabel):
    model =LGBMClassifier()
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]

    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic={}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen']=recall_score(testlabel, prediction)
    dic['f1_score']=f1_score(testlabel, prediction)
    dic['spec']=spec
    dic['mcc']=matthews_corrcoef(testlabel, prediction)
    return dic
from sklearn.naive_bayes import GaussianNB
def bayes(traindata, trainlabel, testdata, testlabel):
    model = GaussianNB()
    model.fit(traindata, trainlabel)
    prediction = model.predict(testdata)
    y_score = model.predict_proba(testdata)
    cm = confusion_matrix(testlabel, prediction)
    print(cm)
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]
    if ((FP + TN) != 0):
        spec = float(float(TN) / float(FP + TN))
    else:
        spec = "error"
    # print((float(TP+TN)/float(TP+FP+TN+FN)))
    print("ACC: %f " % accuracy_score(testlabel, prediction))
    auc = None
    print("F1: %f " % f1_score(testlabel, prediction))
    print("Recall(Sen): %f " % recall_score(testlabel, prediction))
    print("Pre: %f " % precision_score(testlabel, prediction))
    print("MCC: %f " % matthews_corrcoef(testlabel, prediction))
    print("SPEC: %f" % spec)
    dic = {}
    dic['acc'] = accuracy_score(testlabel, prediction)
    dic['sen'] = recall_score(testlabel, prediction)
    dic['f1_score'] = f1_score(testlabel, prediction)
    dic['spec'] = spec
    dic['mcc'] = matthews_corrcoef(testlabel, prediction)
    return dic

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def plotROC(test, score,filename):
    fpr, tpr, threshold = roc_curve(test, score)
    auc_roc = auc(fpr, tpr)
    plt.figure()
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 22,
            }
    lw = 3
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %f)' % auc_roc)
    #    if aucVal is None:
    #        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
    #    else:
    #        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %aucVal)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=20)
    plt.xlabel('False Positive Rate', font)
    plt.ylabel('True Positive Rate', font)
    plt.title('Receiver operating characteristic curve', font)
    plt.legend(loc="lower right")
    filepath = "ROC\\" + filename
    plt.savefig(filepath)
    #plt.show()



def plotPR(test, score,filename):
    precision, recall, thresholds = precision_recall_curve(test, score)
    pr_auc = auc(recall, precision)
    plt.figure()
    lw = 3
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 22,
            }
    plt.figure(figsize=(8, 8))
    plt.plot(precision, recall, color='darkred', lw=lw, label='P-R curve (area = %f)' % pr_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=20)
    plt.xlabel('Recall', font)
    plt.ylabel('Precision', font)
    plt.title('Precision recall curve', font)
    plt.legend(loc="lower right")
    filepath = "PR\\" + filename
    plt.savefig(filepath)
    #plt.show()

def find_best_SVM(c,gamma,traindata,trainlabel,testdata,testlabel):
    model = SVC(C=c, gamma=gamma, probability=True)
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    acc=accuracy_score(testlabel, prediction)
    return (acc,model)

