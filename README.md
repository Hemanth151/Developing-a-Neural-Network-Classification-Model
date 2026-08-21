# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Write your own steps

### STEP 2: 



### STEP 3: 



### STEP 4: 



### STEP 5: 



### STEP 6: 





## PROGRAM

### Name:

### Register Number:

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv("customers.csv")
df

df=df.drop(columns=["ID"])
df

df.fillna({"Work_Experience":0,"Family_Size" :df["Family_Size"].median()},inplace=True)
df

df.columns

cat_columns=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1"]
for col in cat_columns:
 df[col]= LabelEncoder().fit_transform(df[col])

le=LabelEncoder()
df["Segmentation"]=le.fit_transform(df["Segmentation"])
df

x=df.drop(columns=["Segmentation"])
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)

xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)

tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)

class classifier1(nn.Module):
  def __init__(self,input_size):
    super().__init__()
    self.l1=nn.Linear(input_size,32)
    self.l2=nn.Linear(32,16)
    self.l3=nn.Linear(16,8)
    self.l4=nn.Linear(8,4)

  def forward(self,x):
    x=F.relu(self.l1(x))
    x=F.relu(self.l2(x))
    x=F.relu(self.l3(x))
    x=self.l4(x)
    return x

model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)

epochs=100
for i in range(epochs):
  for a,b in trl:
    op.zero_grad()
    pred=model(a)
    loss=criterion(pred,b.long())
    loss.backward()
    op.step()
  if (i%10==0):
    print(f"Loss: {i}/{epochs}",loss.item())

pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
    print(act,pre)

accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy\n",accuracy)
print("Confusion Matrix\n",conf_matrix)
print("Classification Report\n",cl_report)

import seaborn as sns
x1=["A","B","C","D"]
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="Blues",xticklabels=x1,yticklabels=x1)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

sample_input = xst[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = le.inverse_transform([predicted_class_index])[0]
print("Name       : SEERAPU HEMANTH REDDY")
print("Register No: 212225040393")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {le.inverse_transform([int(yst[12].item())])[0]}')




### Dataset Information
<img width="647" height="292" alt="image" src="https://github.com/user-attachments/assets/8fb8d8df-2995-4c66-8197-a0eb3a047768" />



## OUTPUT
<img width="300" height="297" alt="image" src="https://github.com/user-attachments/assets/d24b4dda-f5c8-45d2-8a38-4bf9d31294d4" />









## Confusion Matrix

<img width="287" height="300" alt="image" src="https://github.com/user-attachments/assets/b9511455-d72e-4be5-9eac-29da5c22f7d8" />









## Classification Report
<img width="531" height="490" alt="image" src="https://github.com/user-attachments/assets/81695ca9-d1a9-4548-b7b2-9ca8f3fbda46" />



### New Sample Data Prediction
<img width="312" height="301" alt="image" src="https://github.com/user-attachments/assets/5d061963-8f4d-432e-aec7-7d86b08e15c0" />



Include your sample input and output here

## RESULT

Include your result here
