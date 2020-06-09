def AbsPredictionError(preds, y_test, y_test_buc, num=validate_sample):
    ape=0
    ape_stage = [0,0,0,0]
    ape_bucket = [0,0,0,0,0]
    N_bucket = [0,0,0,0,0]
    
    for i in range(num):
        ape=ape+abs(preds[i]-y_test[i])
    ape=ape/num
    
    k=int(num/4)
    for i in range(k):
        for j in range(4):
            ape_stage[j] = ape_stage[j] + abs(preds[4*i+j]-y_test[4*i+j])
    ape_stage = np.array(ape_stage)        
    ape_stage = ape_stage/num*4
    
    for i in range(num):
        buc = int(y_test_buc[i])
        err = abs(preds[i]-y_test[i])
        ape_bucket[buc] = ape_bucket[buc] + err
        N_bucket[buc] = N_bucket[buc] + 1
    
    for i in range(5):
        ape_bucket[i] = ape_bucket[i]/N_bucket[i]
    return ape, ape_stage, ape_bucket



cur_median_ref = np.array([[17475.78, 8865.5, 4736.5, 2781.0, 1193.5],
                       [80394.89, 37595.45, 21537.35, 13214.63, 5786.07],
                       [52943.97, 39933.12, 30765.57, 6343.04, 398.21],
                       [8653.8, 3015.48, 1505.0, 783.0, 257.88]])
fin_median_ref = np.array([101263.87, 61933.16, 31948.49, 17721.475, 8741.11])


def predict(model1,model2, my_test_data,num=test_sample):
    cur=[]
    final=[]
    cur_buc=[]
    final_buc=[]
    
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
        
        cur_abs = cur_median_ref[i%4][cur_y]
        final_abs = fin_median_ref[final_y]
        
        cur.append(cur_abs)
        final.append(final_abs)
        
        cur_buc.append([cur_y])
        final_buc.append([final_y])
        
    return cur, final, cur_buc, final_buc


def AbsPredictionError_modelcur(cur, cur_buc, y_test, y_test_buc, num=validate_sample):
    ape=0
    ape_stage = [0,0,0,0]
    ape_bucket = [0,0,0,0,0]
    N_bucket = [0,0,0,0,0]
    
    k=int(num/4)
    for i in range(k):
        err = abs(cur[4*i] + cur[4*i+1] + cur[4*i+2] + cur[4*i+3] -y_test[i])
        ape=ape+abs(err)
        buc = int(y_test_buc[4*i])
        #err = abs(preds[i]-y_test[i])
        ape_bucket[buc] = ape_bucket[buc] + err
        N_bucket[buc] = N_bucket[buc] + 1
    
    for i in range(5):
        ape_bucket[i] = ape_bucket[i]/N_bucket[i]
        
    ape=ape/k
    return ape, ape_bucket
