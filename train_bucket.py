import math
import random
import time

import numpy as np
import pandas as pd

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

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda:6" if use_cuda else "cpu")

train_df = pd.read_csv('data/train5_v3.csv')
validate_df = pd.read_csv('data/validate5_v3.csv')
test_df = pd.read_csv('data/test5_v3.csv')

# define the state
feature_fields=['SEX', 'eighty','fifty','old','seventy','sixty','gdp_h','gdp_l', 'gdp_m','east','middle','west','NCMS','OOP','OTHERS',
          'PREFERENCE','UEBMI','URBMI','WELFARE','lin','other','xian','xiao','diag_I','diag_II','diag_III','diag_IV','nstaged','nor','wei_x',
          'yan','CHARLSON_YS','OPERATION','OPERATION_E','DRUG_HL','DRUG_HL_E', 'DRUG_MY','DRUG_MY_E','DRUG_BX','DRUG_BX_E','EXAM','EXAM_E','Z515GX',
          'Z515GX_E','OPERATION_ZQG','OPERATION_ZQG_E','CRITICAL_COND_DAYS','CRITICAL_COND_DAYS_E','SERIOUS_COND_DAYS','SERIOUS_COND_DAYS_E',
          'EMER_TREAT_TIMES','EMER_TREAT_TIMES_E','ESC_EMER','ESC_EMER_TIMES_E','ESC_EMER_RATE','ESC_EMER_RATE_E','ICU_DAYS','ICU_DAYS_E',
          'CCU_DAYS','CCU_DAYS_E','SPEC_LEVEL_NURS_DAYS','SPEC_LEVEL_NURS_DAYS_E','FIRST_LEVEL_NURS_DAYS','FIRST_LEVEL_NURS_DAYS_E',
          'SECOND_LEVEL_NURS_DAYS','SECOND_LEVEL_NURS_DAYS_E','INP_DAYS','INP_DAYS_E','BEFORE_OPERATION','BEFORE_OPERATION_E','BEFOR_QZ',
          'ZJJG1_E','hao','huai','unknown','wei_y','FIRST1','FIRST2','FIRST3','FIRST4','FIRST5','FIRST9','SECOND1','SECOND2','SECOND3',
          'SECOND4','EXCISION1','EXCISION2','EXCISION3','OTHERS2','no','MAJOR PROCEDURE_E',
          'COSTS_MED','COSTS_MED_E','COSTS_MED_RATE','COSTS_TEST','COSTS_TEST_E','COSTS_TEST_RATE','COSTS_EXAM','COSTS_EXAM_E','COSTS_EXAM_RATE',
          'COSTS_TREAT','COSTS_TREAT_E','COSTS_TREAT_RATE','COSTS_OPE','COSTS_OPE_E','COSTS_OPE_RATE','COSTS_MA','COSTS_MA_E','COSTS_MA_RATE',
          'COSTS_BLOOD','COSTS_BLOOD_E','COSTS_BLOOD_RATE','COSTS_MAT','COSTS_MAT_E','COSTS_MAT_RATE','COSTS_ZHEN','COSTS_ZHEN_E',
          'COSTS_ZHEN_RATE','COSTS_OTHER','COSTS_OTHER_E','COSTS1_sum','COSTS1_E','COSTS2_sum','COSTS2_E',
          'COSTS12_sum','COSTS12_E','COSTS3_sum','COSTS3_E','COSTS123_sum','COSTS123_E']

label_fields= ['stage1_y','stage2_y','stage3_y','stage4_y','final_y']
print(len(feature_fields))

dimension_obs=len(feature_fields)
train_sample=train_df.shape[0]
train_num=int((train_df.shape[0])/4)
print(train_num,train_sample,dimension_obs)

class Pseudo_env(object):
    def __init__(self,df,mode="random"):
        self.df=df.copy()
        self.observation_space=dimension_obs # len(feature_fields)
        self.cur_patient=0
        self.cur_id=0
        self.cur_stage=0
        self.mode=mode
        self.cur_state=None
        self.action_space=4 #change in different settings
    
    # may directly choose the first step in every trajectory
    def reset(self):
        if self.mode=="random":
            self.cur_patient=np.random.randint(train_num)
        elif self.mode=="order":
            self.cur_patient=(self.cur_patient+1)%train_num
        self.cur_id = 4*self.cur_patient
        self.cur_stage=0
        self.cur_state=self.df.loc[self.cur_id,feature_fields].values
        return self.cur_state
    
    def step(self):
        
        done=False
        final_action =int(self.df.loc[self.cur_id, 'totalcost'])        
        if self.cur_stage==0:
            current_action =int(self.df.loc[self.cur_id, 'cost1'])
        elif self.cur_stage==1:
            current_action =int(self.df.loc[self.cur_id, 'cost2'])
        elif self.cur_stage==2:
            current_action =int(self.df.loc[self.cur_id, 'cost3'])
        elif self.cur_stage==3:
            current_action =int(self.df.loc[self.cur_id, 'cost4'])
            done=True
        
        self.cur_stage = (self.cur_stage+1)%4
        #reward = self.df.loc[self.cur,'reward']
        next_state = np.zeros(dimension_obs)
            
        if self.cur_stage!=3:
            next_state = self.df.loc[self.cur_id + 1, feature_fields].values
            self.cur_id+=1
        else:
            # trajectory is finished
            next_state = np.zeros(dimension_obs)
        self.cur_state=next_state
        return next_state,done,final_action, current_action
    
    def rollout(self):
        
        if self.mode=="random":
            self.cur_patient=np.random.randint(train_num)
        elif self.mode=="order":
            self.cur_patient=(self.cur_patient+1)%train_num
        patient_id = 4*self.cur_patient
        
        final_action =int(self.df.loc[patient_id, 'totalcost'])
        
        state1 =  self.df.loc[patient_id, feature_fields].values
        state2 =  self.df.loc[patient_id+1, feature_fields].values
        state3 =  self.df.loc[patient_id+2, feature_fields].values
        state4 =  self.df.loc[patient_id+3, feature_fields].values
        
        action1 = int(self.df.loc[patient_id, 'cost1'])
        action2 = int(self.df.loc[patient_id+1, 'cost2'])
        action3 = int(self.df.loc[patient_id+2, 'cost3'])
        action4 = int(self.df.loc[patient_id+3, 'cost4'])
        
        states=np.vstack((state1,state2,state3,state4))
        actions=np.squeeze(np.vstack((action1,action2,action3,action4)))

        return states, actions, final_action

env_name = train_df
env = Pseudo_env(env_name,mode="order")

# We read the dataset and create an iterable.
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

# dataloader.
my_data = my_patients('data/train5_x_v3.csv','data/train5_final_y_v3.csv','data/train5_cur_y_v3.csv')
my_validate_data = my_patients('data/validate5_x_v3.csv','data/validate5_final_y_v3.csv','data/validate5_cur_y_v3.csv')
#my_validate_data = my_patients('data/test5_x_v2.csv','data/test5_final_y_v2.csv','data/test5_cur_y_v2.csv')
batch_size = 64
my_loader = data.DataLoader(my_data,batch_size=batch_size,shuffle=True,num_workers=0)
#my_validate_loader = data.DataLoader(my_validate_data,batch_size=batch_size,num_workers=0)

# simple NN to see its preliminary result
class Model(nn.Module):
    
    def __init__(self, num_inputs=131, hidden=64, num_outputs=5):
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

# the model, the loss function and the optimizer 
# Model.
model1 = Model().to(device)
model2 = Model().to(device)
# Negative log likelihood loss or CEL
criterium = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

# Adam optimizer with learning rate 0.1 and L2 regularization with weight 1e-4.
optimizer1 = torch.optim.Adam(model1.parameters(),lr=0.0001)
optimizer2 = torch.optim.Adam(model2.parameters(),lr=0.0001)

validate_sample = validate_df.shape[0]

def validate(model1,model2,my_validate_data,num=validate_sample):
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
    
    val_model1 = model1
    val_model2 = model2
    val_model1.eval()
    val_model2.eval()
    
    for i in range(num):
        state =  Variable(torch.FloatTensor(np.float32(my_validate_data.tdata[i])),requires_grad=False).to(device)
        cur_value = val_model2(state)
        final_value = val_model1(state)
        
        cur_y = cur_value.sort(descending=True)[1][0].item()
        final_y = final_value.sort(descending=True)[1][0].item()
        
        cur_l = my_validate_data.ctarget[i]
        final_l = my_validate_data.ftarget[i]
        
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

def plot(cur_acc, fin_acc, current1, current2, current3, current4, final1, final2, final3, final4, epoch):
    clear_output(True)
    plt.figure(figsize=(20,18))
    plt.subplot(331)
    plt.title('epoch %s. validate current/final accuracy: %4s/%4s' % (epoch, cur_acc[-1], fin_acc[-1]))
    plt.plot(cur_acc)
    plt.plot(fin_acc)
    
    plt.subplot(332)
    plt.title('epoch %s. validate current accuracy stage1: %4f' % (epoch, current1[-1]))
    plt.plot(current1)
    plt.subplot(333)
    plt.title('epoch %s. validate current accuracy stage2: %4f' % (epoch, current2[-1]))
    plt.plot(current2)
    plt.subplot(334)
    plt.title('epoch %s. validate current accuracy stage3: %4f' % (epoch, current3[-1]))
    plt.plot(current3)
    plt.subplot(335)
    plt.title('epoch %s. validate current accuracy stage4: %4f' % (epoch, current4[-1]))
    plt.plot(current4)
    plt.subplot(336)
    plt.title('epoch %s. validate final accuracy stage1: %4f' % (epoch, final1[-1]))
    plt.plot(final1)
    plt.subplot(337)
    plt.title('epoch %s. validate final accuracy stage2: %4f' % (epoch, final2[-1]))
    plt.plot(final2)
    plt.subplot(338)
    plt.title('epoch %s. validate final accuracy stage3: %4f' % (epoch, final3[-1]))
    plt.plot(final3)
    plt.subplot(339)
    plt.title('epoch %s. validate final accuracy stage4: %4f' % (epoch, final4[-1]))
    plt.plot(final4)
    
    if len(cur_acc)==100:
        plt.savefig('hitrate_pic/=train_bucket.png')
  
    plt.show()

# Taining.
cur=[]
fin=[]
current1=[]
current2=[]
current3=[]
current4=[]
final1=[]
final2=[]
final3=[]
final4=[]

for epoch in range(100):
    for k, (tdata, ftarget,ctarget) in enumerate(my_loader,0):
    # Definition of inputs as variables for the net.
    # requires_grad is set False because we do not need to compute the 
    # derivative of the inputs.
        tdata   = Variable(tdata,requires_grad=False)
        ftarget = Variable(ftarget.long(),requires_grad=False).to(device)
        ctarget = Variable(ctarget.long(),requires_grad=False).to(device)
    
        # Set gradient to 0.
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # Feed forward.
        pred1 = model1(tdata)
        pred2 = model2(tdata)
        # Loss calculation.
        #loss1 = criterium(pred1,ftarget.view(-1))
        #loss2 = criterium(pred2,ctarget.view(-1))
        loss1 = criterion(pred1,ftarget.view(-1))
        loss2 = criterion(pred2,ctarget.view(-1))
        # Gradient calculation.
        loss1.backward()
        loss2.backward()
        # Print loss every 10 iterations.
        if k%100==0:
            print('Loss1 {:.4f} at iter {:d} at epoch {:d}'.format(loss1.item(),k,epoch+1))
            print('Loss2 {:.4f} at iter {:d} at epoch {:d}'.format(loss2.item(),k,epoch+1))
        # Model weight modification based on the optimizer. 
        optimizer1.step()
        optimizer2.step()
        if k==3603:
            #my_loader = data.DataLoader(my_data,batch_size=batch_size,num_workers=0)
            cur_acc, fin_acc, current_acc_stage1,current_acc_stage2,current_acc_stage3,current_acc_stage4, final_acc_stage1,final_acc_stage2,final_acc_stage3,final_acc_stage4 = validate(model1,model2,my_validate_data)
            cur.append(cur_acc)
            fin.append(fin_acc)
            current1.append(current_acc_stage1)
            current2.append(current_acc_stage2)
            current3.append(current_acc_stage3)
            current4.append(current_acc_stage4)
            final1.append(final_acc_stage1)
            final2.append(final_acc_stage2)
            final3.append(final_acc_stage3)
            final4.append(final_acc_stage4)
            plot(cur, fin, current1, current2, current3, current4, final1, final2, final3, final4, epoch+1)
            print('Validate acc for current cost {:.4f} at iter {:d} at epoch {:d}'.format(cur_acc,k,epoch+1))
            print('Validate acc for total cost {:.4f} at iter {:d} at epoch {:d}'.format(fin_acc,k,epoch+1))
            break

torch.save(model1.state_dict(),'model_save/1model_params.pkl')
torch.save(model2.state_dict(),'model_save/2model_params.pkl')

plt.figure(figsize=(20,18))
plt.subplot(331)
plt.title('epoch %s. validate current/final accuracy: %4s/%4s' % (epoch, cur_acc[-1], fin_acc[-1]))
plt.plot(cur_acc)
plt.plot(fin_acc)

