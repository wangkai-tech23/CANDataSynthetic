import numpy as np
from numpy import *
import pandas as pd
import csv

df_iter=pd.read_csv("./car_hacking/csv/chuli/spoofing_the_drive_gear_ttt.csv",iterator=True,chunksize=10000)
o_data=pd.concat([dfi for dfi in df_iter])
o_data=mat(o_data)

data=np.array(o_data)    

heng=data.shape[0]
shu=data.shape[1]    

zz=np.zeros((heng, 96), int)


for i in range(0,heng):    
    a = data[i][0]
    for j in range(15,-1,-1):
        zz[i][j]=a%2
        a=a/2

    a = data[i][1]
    for j in range(31,15,-1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][3]
    for j in range(39,31,-1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][4]
    for j in range(47, 39, -1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][5]
    for j in range(55, 47, -1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][6]
    for j in range(63, 55, -1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][7]
    for j in range(71, 63, -1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][8]
    for j in range(79, 71, -1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][9]
    for j in range(87, 79, -1):
        zz[i][j] = a % 2
        a = a / 2

    a = data[i][10]
    for j in range(95, 87, -1):
        zz[i][j] = a % 2
        a = a / 2




with open('./car_hacking/csv/chuli/spoofing_the_drive_gear_ttt_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    lll_list = np.arange(0, 96, 1)
    writer.writerow(lll_list)
    writer.writerows(zz)

file.close()