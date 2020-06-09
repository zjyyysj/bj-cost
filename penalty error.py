#default import and class
penalty = np.array([[0,2,4,6,8],[1,0,2,4,6],[2,1,0,2,4],[3,2,1,0,2],[4,3,2,1,0]])

def inference_penalty(model1,model2, my_test_data,num=test_sample,penalty = penalty):
    #pe: penalty error 
    cur_pe=0
    final_pe=0

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
        
        cur_l = my_test_data.ctarget[i]
        final_l = my_test_data.ftarget[i]
        
        cur_pe = cur_pe + penalty[int(cur_y),int(cur_l)]
        final_pe = final_pe + penalty[int(final_y),int(final_l)]
    
    return cur_pe/num, final_pe/num

def inference_penalty_stage(model1,model2, my_test_data,num=test_sample,penalty = penalty):
    #pe: penalty error 
    cur_pe=np.zeros(4)
    final_pe=np.zeros(4)
    k=int(num/4)

    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    for i in range(k):
        for j in range(4):
            state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[4*i+j])),requires_grad=False).to(device)
            cur_value = test_model2(state)
            final_value = test_model1(state)
            
            cur_y = cur_value.sort(descending=True)[1][0].item()
            final_y = final_value.sort(descending=True)[1][0].item()
            
            cur_l = my_test_data.ctarget[4*i+j]
            final_l = my_test_data.ftarget[4*i+j]
            
            cur_pe[j] = cur_pe[j] + penalty[int(cur_y),int(cur_l)]
            final_pe[j] = final_pe[j] + penalty[int(final_y),int(final_l)]
    
    for j in range(4):
        cur_pe[j] = cur_pe[j]/k
        final_pe[j] = final_pe[j]/k
        
    return cur_pe, final_pe

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

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

sgdc = SGDClassifier()
sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)

from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train,y_train)
lsvc_y_predict = lsvc.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict = dtc.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict = gbc.predict(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,20)) 
mlp.fit(X_train,y_train) 
mlp_y_predict = mlp.predict(X_test)

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train,y_train) 
xgbc_y_predict = xgbc.predict(X_test)
xgbc_error = inference_BLpenalty(y_test, xgbc_y_predict)

lr_error = inference_BLpenalty(y_test, lr_y_predict)
sgdc_error = inference_BLpenalty(y_test, sgdc_y_predict)
lsvc_error = inference_BLpenalty(y_test, lsvc_y_predict)
dtc_error = inference_BLpenalty(y_test, dtc_y_predict)
rfc_error = inference_BLpenalty(y_test, rfc_y_predict)
gbc_error = inference_BLpenalty(y_test, gbc_y_predict)
mlp_error = inference_BLpenalty(y_test, mlp_y_predict)
print(lr_error,sgdc_error,lsvc_error,dtc_error,rfc_error,gbc_error,mlp_error)

[cur_errors, final_errors] = inference_penalty_stage(model1,model2, my_validate_data,validate_sample)
[tcur_errors, tfinal_errors] = inference_penalty_stage(model1,model2, my_test_data)

