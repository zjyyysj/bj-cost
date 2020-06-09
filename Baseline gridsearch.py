import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

train_df = pd.read_csv('data/train5_v2.csv')
validate_df = pd.read_csv('data/validate5_v2.csv')
test_df = pd.read_csv('data/test5_v2.csv')

#my_data = my_patients('data/train5_x_v2.csv','data/train5_final_y_v2.csv','data/train5_cur_y_v2.csv')
X_train = pd.read_csv('data/train5_x_v2.csv').values
y_train = pd.read_csv('data/train5_final_y_v2.csv').values
X_test = pd.read_csv('data/validate5_x_v2.csv').values
y_test = pd.read_csv('data/validate5_final_y_v2.csv').values
#X_test = pd.read_csv('data/test5_x_v2.csv').values
#y_test = pd.read_csv('data/test5_final_y_v2.csv').values

test_sample = validate_df.shape[0]
validate_sample = validate_df.shape[0]

def inference(y_test,y_predict, num=test_sample):
    acc=0
    acc_stage = np.zeros(4)   
    
    for i in range(num):
        
        label = y_test[i]
        predict = y_predict[i]
        k = i%4
        
        if label == predict:
            acc = acc + 1
            acc_stage[k] = acc_stage[k] + 1   
            
    acc = acc/num
    acc_stage = acc_stage/num*4
    
    return acc, acc_stage

lr = LogisticRegression()
penalty =['l1','l2']
C = np.logspace(0,4,10)
hyperparameters = dict(C=C, penalty=penalty)

clf = GridSearchCV(lr, hyperparameters, cv=5, verbose=0,n_jobs=-1)
best_model = clf.fit(X_train, y_train)

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

lr_y_predict = best_model.predict(X_test)
print('Accruacy of LR Classifier:',best_model.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['1','2','3','4','5']))

inference(y_test, lr_y_predict)

disp = metrics.plot_confusion_matrix(best_model, X_test, y_test)
disp.figure_.suptitle('Confusion Matrix')
print('Confusion matrix:\n%s' % disp.confusion_matrix)

penalty = np.array([[0,2,4,6,8],[1,0,2,4,6],[2,1,0,2,4],[3,2,1,0,2],[4,3,2,1,0]])
penalty

def inference_BLpenalty(y_test, y_predict, num=validate_sample,penalty = penalty):
    #pe: penalty error 
    pe=0
    
    for i in range(num):
        
        pe = pe + penalty[int(y_predict[i]),int(y_test[i])]
    
    return pe/num

def inference_BLpenalty_stage(y_test, y_predict, num=validate_sample,penalty = penalty):
    #pe: penalty error 
    pe=np.zeros(4)
    k=int(num/4)
    
    for i in range(k):
        for j in range(4):
            pe[j] = pe[j] + penalty[int(y_predict[4*i+j]),int(y_test[4*i+j])]
    
    for j in range(4):
        pe[j] = pe[j]/k
        
    return pe

