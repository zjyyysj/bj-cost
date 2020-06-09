#default import
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

train_df = pd.read_csv('data/train4_v3.csv')
validate_df = pd.read_csv('data/validate4_v3.csv')
test_df = pd.read_csv('data/test4_v3.csv')
whole_df = pd.read_csv('data/whole4_v3.csv')

X_train = pd.read_csv('data/train4_x_v3.csv').values
y_train = pd.read_csv('data/train4_final_y_v3.csv').values
X_test = pd.read_csv('data/validate4_x_v3.csv').values
y_test = pd.read_csv('data/validate4_final_y_v3.csv').values

X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)
X.shape[0]

train_sample = train_df.shape[0]
N = X.shape[0]
test_sample = validate_df.shape[0]

my_train_data = my_patients('data/train4_x_v3.csv','data/train4_final_y_v3.csv','data/train4_cur_y_v3.csv')
my_test_data = my_patients('data/test4_x_v3.csv','data/test4_final_y_v3.csv','data/test4_cur_y_v3.csv')
my_validate_data = my_patients('data/validate4_x_v3.csv','data/validate4_final_y_v3.csv','data/validate4_cur_y_v3.csv')

def predict(model1,model2, my_test_data,num=test_sample):
    cur=[]
    final=[]
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    for i in range(num):
        state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[i])),requires_grad=False).to(device)
        cur_value = test_model2(state)
        final_value = test_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur.append([cur_y])
        final.append([final_y])
        
    return cur, final

lr = LogisticRegression()
penalty =['l1','l2']
C = np.logspace(0,4,10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(lr, hyperparameters, cv=5, verbose=0,n_jobs=-1)
best_model = clf.fit(X[::4], final[::4])
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

best_model.score(X[::4], final[::4])

#调用LogisticRegression的fit函数来训练模型,并使用predict函数进行预测
lr.fit(X[::4],final[::4])
#lr_y_predict = lr.predict(X_test)
print('Accruacy of LR Classifier:',lr.score(X[::4],final[::4]))

# identify low bucket
m={}
n={}
for i in range(len(col_name)):
    m.setdefault(col_name[i],0)
    n.setdefault(col_name[i],0)
    m[col_name[i]]=abs(lr.coef_[3][i])
    n[col_name[i]]=lr.coef_[3][i]
    
sorted(m.items(), key=lambda x:x[1], reverse=True)

feature_fields5 = ['Medicine cost','Treatment cost','Other cost', 'Material cost','Operation','Length of stay',
                  'Days before diagnosis', 'Test cost', 'Diagnosis cost','Nursing days in second level'
                  , 'Blood cost']
feature_fields2 = ['Medicine cost','Treatment cost','Operation', 'Material cost','Days before diagnosis',
                   'Length of stay','Nursing days in first level','Immunotherapy',
                   'Diagnosis cost', 'Test cost','Bronchial operation', 'Other cost']
feature_fields5 = ['Medicine cost','Treatment cost','Other cost', 'Material cost','Operation','Length of stay',
                  'Days before diagnosis', 'Test cost', 'Diagnosis cost','Nursing days in second level'
                  , 'Blood cost']

figsize=(24,12)
fig,ax=plt.subplots(figsize=figsize)
ax.barh(feature_fields5, importances5)
labels = ax.get_xticklabels()
plt.setp(labels, rotation = 45, horizontalalignment='right')
ax.set(xlim=[0,21])
ax.set_title('a',x=-0.1,fontsize=30)
#ax.axvline(importances_cli.mean(), ls='--', color='r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel(xlabel='Coefficients weight for bucket 5 patients (extremely high)',fontsize=25)
ax.set_ylabel(ylabel='Top Coefficients weights',fontsize=25)

feature_fields1 = ['Medicine cost','Material cost','Length of stay', 'Diagnosis cost','Other cost','Nursing days in second level',
                  'Nursing days in first level', 'Test cost','Days before diagnosis', 'Treatment cost','Immunotherapy','Examination cost']

feature_fields4 = ['Medicine cost','Material cost','Diagnosis cost','Immunotherapy','Length of stay', 'Treatment cost','Other cost','Nursing days in second level',
                  'Operation', 'Days before diagnosis', 'Exam','Nursing days in first level']

feature_fields3 = ['Material cost','Treatment cost','Diagnosis cost','Length of stay','Operation','Nursing days in first level',
                   'Nursing days in second level','Medicine cost','Operation cost','Anesthesia cost','ICU Days', 'Blood cost']


