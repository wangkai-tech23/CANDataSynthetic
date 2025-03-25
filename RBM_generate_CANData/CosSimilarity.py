import numpy as np
import pandas as pd
import csv

def mo(a):
    n=len(a)
    ans=0
    for i in range(0,n):
        ans=ans+a[i]

    ans=np.sqrt(ans)
    return ans


def neiji(a,b):
    n=len(a)
    ans = 0
    for i in range(0, n):
        ans = ans + a[i]*b[i]

    return ans


def cos(a,b):
    aa=mo(a)
    bb=mo(b)
    if aa==0:
        aa=1
    if bb==0:
        bb=1
    cc=neiji(a,b)
    ans=cc/(aa*bb)

    return ans


#a和b需为array
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm==0:
        a_norm=1
    if b_norm==0:
        b_norm=1
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos


choose_lines=8000

data1=pd.read_csv("./car_hacking/csv/chuli/dos_ttt_2.csv",error_bad_lines=False,nrows=choose_lines)
data1=np.array(data1)     
data2=pd.read_csv("./gen_data_rbm/gen_dos_ttt.csv",error_bad_lines=False,nrows=choose_lines)
data2=np.array(data2)     



heng=data2.shape[0]
shu=data2.shape[1]     


zz=np.zeros(heng, float)


for i in range(0,heng):    
    zz[i]=cos(data1[i],data2[i])

ans=0
for i in range(0,heng):    
    ans=ans+zz[i]
ans=ans/heng
print(ans)
print(zz)