import numpy as np
import pandas as pd
import csv

data=pd.read_csv("./gen_data_rbm/another_gen_spoof_RPM_ttt.csv",error_bad_lines=False)
data=np.array(data)    

heng=data.shape[0]
shu=data.shape[1]   

zz=np.zeros((heng, 9), int)


for i in range(0,heng):    

    a = 0
    b = 0
    for j in range(31,15,-1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][0] = a

    a = 0
    b = 0
    for j in range(39,31,-1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][1] = a

    a = 0
    b = 0
    for j in range(47, 39, -1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][2] = a

    a = 0
    b = 0
    for j in range(55, 47, -1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][3] = a

    a = 0
    b = 0
    for j in range(63, 55, -1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][4] = a

    a = 0
    b = 0
    for j in range(71, 63, -1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][5] = a

    a = 0
    b = 0
    for j in range(79, 71, -1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][6] = a

    a = 0
    b = 0
    for j in range(87, 79, -1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][7] = a

    a = 0
    b = 0
    for j in range(95, 87, -1):
        a=a+(2**b)*data[i][j]
        b = b + 1
    zz[i][8] = a



#写入数据
with open('./gen_data_rbm/another_gen_spoof_RPM_ttt_10_10.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    lll_list = np.arange(0, 11, 1)
    writer.writerow(lll_list)
    writer.writerows(zz)

file.close()