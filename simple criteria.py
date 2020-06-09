#default import and class
#70 before QZ(max 20days) 18 welfare 6 gdp_h, 116 costs_zhen, 9 east, 19 lin, 21 xian, 2 fifty, 3 old, 5 sixty
#95 costs_test, 92 medicine cost, 113 material cost, 66 length of stay (max 90 days)
#32 operation 34 HL 36 MY 38 BX 40 EXAM 42 GX 44 ZQG
#64 secondary care (max 70) 62 first care (max60)
#12 'NCMS','OOP','OTHERS','PREFERENCE','UEBMI','URBMI','WELFARE'
#23 stage I 24 stage II 8 gdp_m 29 wei_x 30 yan 75 wei_y, 31 charlson
# 0 sex 1 70-80 10 middle 22 xiao 26 stage IV
def threshold(pa, item1=26, item2=26, item3=22, item4=22, item5=10,item6=10):
    thr = 0
    #num = 0
    if pa[item1]+pa[item2]==2:
        if pa[item3]+pa[item4]>0:
            if pa[item5]+pa[item6]==2:
                thr=1
                #num=num+1
    return thr    
    
def predict_threshold(model1,model2, my_test_data,num=test_sample, bucket=[0]):
    final = []
    thr_num = 0
    pre_num = 0
    act_num = 0
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    for i in range(num):
        pa = my_test_data.tdata[i]
        f = my_test_data.ftarget[i]
        #c = my_test_data.ctarget[i]
        
        if threshold(pa)>0:
            thr_num = thr_num + 1
            state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[i])),requires_grad=False).to(device)
            #cur_value = test_model2(state)
            final_value = test_model1(state)
        
            #cur_y = cur_value.sort(descending=True)[1][0].item()
            final_y = final_value.sort(descending=True)[1][0].item()
            if final_y in bucket:
                pre_num = pre_num + 1
            if f in bucket:
                act_num = act_num + 1
            #cur.append([cur_y])
            #final.append([final_y])
    if thr_num==0:
        return 0,0,0
    else:
        return thr_num/num, pre_num/thr_num, act_num/thr_num

predict_threshold(model1,model2, my_validate_data,num=validate_sample)

#70 before QZ(max 20days) 18 welfare 6 gdp_h, 116 costs_zhen, 
#95 costs_test, 92 medicine cost, 113 material cost, 66 length of stay (max 90 days)
#25 diag III, 26 diag IV, 101 treatment cost, 32 operation
def threshold_(pa, item1=66, item2=32, item3=92,bucket=[0.5,0.75,1]):
    thr = 0
    #num = 0
    if pa[item1] >=0.07:
        if pa[item2] == 1:
            if pa[item3] in bucket:
                thr=1
                #num=num+1
    return thr    
    
def predict_threshold_(model1,model2, my_test_data,num=test_sample, bucket=[3,4]):
    final = []
    thr_num = 0
    pre_num = 0
    act_num = 0
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    for i in range(num):
        pa = my_test_data.tdata[i]
        f = my_test_data.ftarget[i]
        #c = my_test_data.ctarget[i]
        
        if threshold_(pa)==1:
            thr_num = thr_num + 1
            state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[i])),requires_grad=False).to(device)
            #cur_value = test_model2(state)
            final_value = test_model1(state)
        
            #cur_y = cur_value.sort(descending=True)[1][0].item()
            final_y = final_value.sort(descending=True)[1][0].item()
            if final_y in bucket:
                pre_num = pre_num + 1
            if f in bucket:
                act_num = act_num + 1
            #cur.append([cur_y])
            #final.append([final_y])
        
    return thr_num/num, pre_num/thr_num, act_num/thr_num


[a2,b2,c2]=predict_threshold_(model1,model2, my_validate_data,validate_sample)

