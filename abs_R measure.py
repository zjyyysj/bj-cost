def AbsR_measure(preds, y_test, y_test_buc, num=validate_sample, median = l, 
                 bucket_median = [10264,21600,40728,74148,160784]):
    up = 0
    down = 0
    
    up_stage = [0,0,0,0]
    down_stage = [0,0,0,0]
    R_stage=[0,0,0,0]
    
    #bucket_median = [10264,21600,40728,74148,160784]
    #bucket_median = [l,l,l,l,l]
    up_bucket=[0,0,0,0,0]
    down_bucket=[0,0,0,0,0]
    R_bucket=[0,0,0,0,0]
    
    for i in range(num):
        down = down +abs(y_test[i]-median)
        
    for i in range(num):
        up = up +abs(preds[i]-median)
    AbsR = 1 - up/down
    
    k=int(num/4)
    for i in range(k):
        for j in range(4):
            down_stage[j] = down_stage[j] + abs(y_test[4*i+j]-median)
            up_stage[j] = up_stage[j] + abs(preds[4*i+j]-y_test[4*i+j])
    
    for i in range(4):
        R_stage[i] = 1-up_stage[i]/down_stage[i]
    
    for i in range(num):
        buc = int(y_test_buc[i])
        down_bucket[buc] = down_bucket[buc] + abs(y_test[i]- bucket_median[buc])
        up_bucket[buc]   = up_bucket[buc]   + abs(preds[i] - y_test[i])
    
    for i in range(5):
        R_bucket[i] = 1-up_bucket[i]/down_bucket[i]
        
    rate=down_bucket[4]/down
    
    return AbsR, R_stage, R_bucket,rate


def R_square_measure(preds, y_test, y_test_buc, num=validate_sample, median = l,
                     bucket_median = [10264,21600,40728,74148,160784]):
    up = 0
    down = 0
    
    up_stage = [0,0,0,0]
    down_stage = [0,0,0,0]
    R_stage = [0,0,0,0]
    
    #bucket_median = [10264,21600,40728,74148,160784]
    #bucket_median = [l,l,l,l,l]
    up_bucket=[0,0,0,0,0]
    down_bucket=[0,0,0,0,0]
    R_bucket = [0,0,0,0,0]
    
    for i in range(num):
        down = down +abs(y_test[i]-median)**2
        
    for i in range(num):
        up = up +abs(preds[i]-median)**2
    AbsR = 1 - up/down
    
    k=int(num/4)
    for i in range(k):
        for j in range(4):
            down_stage[j] = down_stage[j] + (y_test[4*i+j]-median)**2
            up_stage[j] = up_stage[j] + (preds[4*i+j]-y_test[4*i+j])**2
    
    for i in range(4):
        R_stage[i] = 1-up_stage[i]/down_stage[i]
    
    for i in range(num):
        buc = int(y_test_buc[i])
        down_bucket[buc] = down_bucket[buc] + (y_test[i]- bucket_median[buc])**2
        up_bucket[buc]   = up_bucket[buc]   + (preds[i] - y_test[i])**2
    
    for i in range(5):
        R_bucket[i] = 1-up_bucket[i]/down_bucket[i]
    
    rate=down_bucket[4]/down
    return AbsR, R_stage, R_bucket,rate



def AbsR_measure_cur(preds, y_test, num=validate_sample, median = cur_median):
    up = 0
    down = 0
    
    up_stage = [0,0,0,0]
    down_stage = [0,0,0,0]
    R_stage = [0,0,0,0]
    
    for i in range(num):
        down = down +abs(y_test[i]-median[i%4])
        
    for i in range(num):
        up = up +abs(preds[i]-median[i%4])
    AbsR = 1 - up/down
    
    k=int(num/4)
    for i in range(k):
        for j in range(4):
            down_stage[j] = down_stage[j] + abs(y_test[4*i+j]-median[j])
            up_stage[j] = up_stage[j] + abs(preds[4*i+j]-median[j])
            
    for j in range(4):
        R_stage[j] = 1 - up_stage[j]/down_stage[j]
    
    return AbsR, R_stage
