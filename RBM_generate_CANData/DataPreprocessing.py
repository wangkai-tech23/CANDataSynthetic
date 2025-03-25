import numpy as np
from numpy import *
import pandas as pd
import csv

df_iter=pd.read_csv("./car_hacking/csv/Spoofing_the_drive_gear_dataset.csv",iterator=True,chunksize=10000)
o_data=pd.concat([dfi for dfi in df_iter])
o_data=mat(o_data)

o_data=np.array(o_data)  


heng=o_data.shape[0]
shu=o_data.shape[1]

list_del=[]
for i in range(0,heng):
    print(i)
    if o_data[i][2] != 8:
        list_del.append(i)
    elif o_data[i][11]=='R':
        list_del.append(i)

data=np.delete(o_data,list_del,axis=0)


heng=data.shape[0]
shu=data.shape[1]


for i in range(0,heng):  
    data[i][1]=int(data[i][1],16)   
    for j in range(3,11):
        data[i][j]=int(data[i][j],16)  


for i in range(heng-1,0,-1):   
    data[i][0]=int((data[i][0]-data[i-1][0])*1000000)
data[0][0]=0                




with open('./car_hacking/csv/chuli/spoofing_the_drive_gear_ttt.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    list=np.arange(0,12,1)
    writer.writerow(list)
    writer.writerows(data)

file.close()