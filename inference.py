import math
import random
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from PIL import Image
from torch.autograd import Variable
import torch.utils.data as data

from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda:6" if use_cuda else "cpu")

# simple NN to see its preliminary result
class Model(nn.Module):
    
    def __init__(self, num_inputs=131, hidden=32, num_outputs=5):
        super(Model, self).__init__()
        self.fc0 = nn.Linear(num_inputs, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, num_outputs)
        self.logprob = nn.LogSoftmax(dim=1)  

    def forward(self, x):
        x = x.type_as(self.fc0.bias)
        x = F.relu(self.fc0(x))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #x = self.logprob(x)
        return x  

model1 = Model().to(device)
model2 = Model().to(device)
model1.load_state_dict(torch.load('model_save/model_pointc.pkl')) 
model2.load_state_dict(torch.load('model_save/model_pointf.pkl'))

class my_patients(data.Dataset):
    def __init__(self, filename1,filename2,filename3):
        # Read data file.
        self.tdata = pd.read_csv(filename1).values   # x
        self.ftarget = pd.read_csv(filename2).values# label1
        self.ctarget = pd.read_csv(filename3).values# label2
        self.n_samples = self.tdata.shape[0]
    
    def __len__(self):   # Length of the dataset.
        return self.n_samples
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.tdata[index]).to(device), torch.Tensor(self.ftarget[index]).to(device), torch.Tensor(self.ctarget[index]).to(device)

my_train_data = my_patients('data/train5_x_v3.csv','data/train5_final_y_v3.csv','data/train5_cur_y_v3.csv')
my_test_data = my_patients('data/test5_x_v3.csv','data/test5_final_y_v3.csv','data/test5_cur_y_v3.csv')
my_validate_data = my_patients('data/validate5_x_v3.csv','data/validate5_final_y_v3.csv','data/validate5_cur_y_v3.csv')

df = pd.read_csv('data/train5_v3.csv')
train_sample = df.shape[0]
test_df = pd.read_csv('data/test5_v3.csv')
test_sample = test_df.shape[0]
validate_df = pd.read_csv('data/validate5_v3.csv')
validate_sample = validate_df.shape[0]

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

def prob(value, l=5):
    v=value.detach().numpy()
    p=[]
    sum=0
    for i in range(l):
        sum=sum+np.exp(v[i])
    for i in range(l):
        p.append(np.exp(v[i])/sum)
    return p

def inference(model1,model2, my_test_data,num=test_sample):
    current_acc=0
    final_acc=0
    current_acc_stage1=0
    current_acc_stage2=0
    current_acc_stage3=0
    current_acc_stage4=0
    final_acc_stage1=0
    final_acc_stage2=0
    final_acc_stage3=0
    final_acc_stage4=0
    
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
        
        if cur_y == cur_l:
            current_acc = current_acc + 1
        if final_y == final_l:
            final_acc = final_acc + 1
        
        if i%4==0:
            if cur_y == cur_l:
                current_acc_stage1 = current_acc_stage1  + 1
            if final_y == final_l:
                final_acc_stage1  = final_acc_stage1  + 1
        
        if i%4==1:
            if cur_y == cur_l:
                current_acc_stage2 = current_acc_stage2  + 1
            if final_y == final_l:
                final_acc_stage2  = final_acc_stage2  + 1
        
        if i%4==2:
            if cur_y == cur_l:
                current_acc_stage3 = current_acc_stage3  + 1
            if final_y == final_l:
                final_acc_stage3  = final_acc_stage3  + 1
                
        if i%4==3:
            if cur_y == cur_l:
                current_acc_stage4 = current_acc_stage4  + 1
            if final_y == final_l:
                final_acc_stage4  = final_acc_stage4  + 1
            
            
    cur_acc = current_acc/num
    fin_acc = final_acc/num
    current_acc_stage1 = current_acc_stage1/num*4
    current_acc_stage2 = current_acc_stage2/num*4
    current_acc_stage3 = current_acc_stage3/num*4
    current_acc_stage4 = current_acc_stage4/num*4
    final_acc_stage1 = final_acc_stage1/num*4
    final_acc_stage2 = final_acc_stage2/num*4
    final_acc_stage3 = final_acc_stage3/num*4
    final_acc_stage4 = final_acc_stage4/num*4
    
    return cur_acc, fin_acc, current_acc_stage1,current_acc_stage2,current_acc_stage3,current_acc_stage4,final_acc_stage1,final_acc_stage2,final_acc_stage3,final_acc_stage4

inference(model1,model2, my_validate_data, num=validate_sample)

def inference_bucket(model1,model2, my_test_data,num=test_sample, bucket=5):

    current_acc_buc=np.zeros(bucket)
    final_acc_buc=np.zeros(bucket)
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    cur_l_num = np.zeros(bucket)
    final_l_num = np.zeros(bucket)
    
    for i in range(num):
        state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[i])),requires_grad=False).to(device)
        cur_value = test_model2(state)
        final_value = test_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur_l = my_test_data.ctarget[i]
        final_l = my_test_data.ftarget[i]
        
        for j in range(bucket):
            
            if cur_l == j:
                cur_l_num[j] = cur_l_num[j] + 1
                if cur_y == j:
                    current_acc_buc[j] = current_acc_buc[j] + 1
                    
            if final_l == j:
                final_l_num[j] = final_l_num[j] + 1
                if final_y == j:
                    final_acc_buc[j] = final_acc_buc[j] + 1
        
    for k in range(bucket):
        current_acc_buc[k] = current_acc_buc[k]/cur_l_num[k]
        final_acc_buc[k] = final_acc_buc[k]/final_l_num[k]
    
    return current_acc_buc, final_acc_buc, cur_l_num, final_l_num

def inference_bucket_stage1(model1,model2, my_test_data,num=test_sample, bucket=5):

    current_acc_buc=np.zeros(bucket)
    final_acc_buc=np.zeros(bucket)
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    cur_l_num = np.zeros(bucket)
    final_l_num = np.zeros(bucket)
    
    stage_num = int(num/4)
    
    for i in range(stage_num):
        state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[4*i])),requires_grad=False).to(device)
        cur_value = test_model2(state)
        final_value = test_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur_l = my_test_data.ctarget[4*i]
        final_l = my_test_data.ftarget[4*i]
        
        for j in range(bucket):
            
            if cur_l == j:
                cur_l_num[j] = cur_l_num[j] + 1
                if cur_y == j:
                    current_acc_buc[j] = current_acc_buc[j] + 1
                    
            if final_l == j:
                final_l_num[j] = final_l_num[j] + 1
                if final_y == j:
                    final_acc_buc[j] = final_acc_buc[j] + 1
        
    for k in range(bucket):
        current_acc_buc[k] = current_acc_buc[k]/cur_l_num[k]
        final_acc_buc[k] = final_acc_buc[k]/final_l_num[k]
    
    return current_acc_buc, final_acc_buc, cur_l_num, final_l_num

def inference_bucket_stage2(model1,model2, my_test_data,num=test_sample, bucket=5):

    current_acc_buc=np.zeros(bucket)
    final_acc_buc=np.zeros(bucket)
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    cur_l_num = np.zeros(bucket)
    final_l_num = np.zeros(bucket)
    
    stage_num = int(num/4)
    
    for i in range(stage_num):
        state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[4*i+1])),requires_grad=False).to(device)
        cur_value = test_model2(state)
        final_value = test_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur_l = my_test_data.ctarget[4*i+1]
        final_l = my_test_data.ftarget[4*i+1]
        
        for j in range(bucket):
            
            if cur_l == j:
                cur_l_num[j] = cur_l_num[j] + 1
                if cur_y == j:
                    current_acc_buc[j] = current_acc_buc[j] + 1
                    
            if final_l == j:
                final_l_num[j] = final_l_num[j] + 1
                if final_y == j:
                    final_acc_buc[j] = final_acc_buc[j] + 1
        
    for k in range(bucket):
        current_acc_buc[k] = current_acc_buc[k]/cur_l_num[k]
        final_acc_buc[k] = final_acc_buc[k]/final_l_num[k]
    
    return current_acc_buc, final_acc_buc, cur_l_num, final_l_num

def inference_bucket_stage3(model1,model2, my_test_data,num=test_sample, bucket=5):

    current_acc_buc=np.zeros(bucket)
    final_acc_buc=np.zeros(bucket)
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    cur_l_num = np.zeros(bucket)
    final_l_num = np.zeros(bucket)
    
    stage_num = int(num/4)
    
    for i in range(stage_num):
        state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[4*i+2])),requires_grad=False).to(device)
        cur_value = test_model2(state)
        final_value = test_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur_l = my_test_data.ctarget[4*i+2]
        final_l = my_test_data.ftarget[4*i+2]
        
        for j in range(bucket):
            
            if cur_l == j:
                cur_l_num[j] = cur_l_num[j] + 1
                if cur_y == j:
                    current_acc_buc[j] = current_acc_buc[j] + 1
                    
            if final_l == j:
                final_l_num[j] = final_l_num[j] + 1
                if final_y == j:
                    final_acc_buc[j] = final_acc_buc[j] + 1
        
    for k in range(bucket):
        current_acc_buc[k] = current_acc_buc[k]/cur_l_num[k]
        final_acc_buc[k] = final_acc_buc[k]/final_l_num[k]
    
    return current_acc_buc, final_acc_buc, cur_l_num, final_l_num

def inference_bucket_stage4(model1,model2, my_test_data,num=test_sample, bucket=5):

    current_acc_buc=np.zeros(bucket)
    final_acc_buc=np.zeros(bucket)
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    cur_l_num = np.zeros(bucket)
    final_l_num = np.zeros(bucket)
    
    stage_num = int(num/4)
    
    for i in range(stage_num):
        state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[4*i+3])),requires_grad=False).to(device)
        cur_value = test_model2(state)
        final_value = test_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur_l = my_test_data.ctarget[4*i+3]
        final_l = my_test_data.ftarget[4*i+3]
        
        for j in range(bucket):
            
            if cur_l == j:
                cur_l_num[j] = cur_l_num[j] + 1
                if cur_y == j:
                    current_acc_buc[j] = current_acc_buc[j] + 1
                    
            if final_l == j:
                final_l_num[j] = final_l_num[j] + 1
                if final_y == j:
                    final_acc_buc[j] = final_acc_buc[j] + 1
        
    for k in range(bucket):
        current_acc_buc[k] = current_acc_buc[k]/cur_l_num[k]
        final_acc_buc[k] = final_acc_buc[k]/final_l_num[k]
    
    return current_acc_buc, final_acc_buc, cur_l_num, final_l_num

inference_bucket(model1,model2, my_validate_data)

inference_bucket_stage1(model1,model2, my_validate_data)

def inference_y(model1,model2, my_test_data, num=test_sample):
    
    test_model1 = model1
    test_model2 = model2
    test_model1.eval()
    test_model2.eval()
    
    cur_predict = np.zeros(num)
    final_predict = np.zeros(num)
    
    for i in range(num):
        state =  Variable(torch.FloatTensor(np.float32(my_test_data.tdata[i])),requires_grad=False).to(device)
        cur_value = test_model2(state)
        final_value = test_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur_predict[i] = cur_y
        final_predict[i] = final_y
    
    return cur_predict, final_predict

def inference_stage(y_test,y_predict, num=test_sample):
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

cur_predict, final_predict = inference_y(model1,model2,my_test_data)
