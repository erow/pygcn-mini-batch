#%%
import numpy as np
C = 1
F = 2
H =3
y_true = [C,C,C,C,C,C, F, F, F, F, F, F, F, F, F, F, H,H,H,H,H,H,H,H,H]
y_pred = [C,C,C,C,H, F, C,C,C,C,C,C,H,H, F, F, C,C,C,H,H,H,H,H,H]

y_true, y_pred = np.array(y_true), np.array(y_pred)
y_true
# %%
def precision(TP,FP,FN):
    return TP/(TP+FP)

def recall(TP,FP,FN):
    return TP/(TP+FN)

def f1_score(TP,FP,FN):
    p = precision(TP,FP,FN)
    r = recall(TP,FP,FN)
    return 2 * p * r / (r + p)

def micro_f1(y_true,y_pred):
    TP_s,FP_s,FN_s = 0,0,0
    for c in range(1,4):
        TP = (y_true == c) &  (c == y_pred)
        FN = (y_true == c) &  (c != y_pred)
        FP = (y_true != c) &  (c == y_pred)
        TP,FP,FN = (np.sum(i) for i in (TP,FP,FN))
        TP_s,FP_s,FN_s = map(np.sum,
            zip((TP_s,FP_s,FN_s),(TP,FP,FN)))

    return f1_score(TP_s,FP_s,FN_s)

def macro_f1(y_true,y_pred):
    conditional_f1 = []
    for c in range(1,4):
        TP = (y_true == c) &  (c == y_pred)
        FN = (y_true == c) &  (c != y_pred)
        FP = (y_true != c) &  (c == y_pred)
        TP,FP,FN = (np.sum(i) for i in (TP,FP,FN))
        conditional_f1.append(f1_score(TP,FP,FN))
    return np.mean(conditional_f1)

# %% micro 

TP_s,FP_s,FN_s = 0,0,0
for i in range ( 0 , 25 , 5 ) :
    mini_batch_y_true = y_true[ i : i +5]
    mini_batch_y_pred = y_pred[ i : i +5]

    for c in range(1,4):
        TP = (mini_batch_y_true == c) &  (c == mini_batch_y_pred)
        FN = (mini_batch_y_true == c) &  (c != mini_batch_y_pred)
        FP = (mini_batch_y_true != c) &  (c == mini_batch_y_pred)
        TP,FP,FN = (np.sum(i) for i in (TP,FP,FN))
        TP_s,FP_s,FN_s = map(np.sum,
            zip((TP_s,FP_s,FN_s),(TP,FP,FN)))


micro_f1(y_true,y_pred),f1_score(TP_s,FP_s,FN_s)

# %% macro 
from collections import defaultdict
TP_l,FP_l,FN_l =defaultdict(int),defaultdict(int),defaultdict(int)

for i in range ( 0 , 25 , 5 ) :
    mini_batch_y_true = y_true[ i : i +5]
    mini_batch_y_pred = y_pred[ i : i +5]

    for c in range(1,4):
        TP = (mini_batch_y_true == c) &  (c == mini_batch_y_pred)
        FN = (mini_batch_y_true == c) &  (c != mini_batch_y_pred)
        FP = (mini_batch_y_true != c) &  (c == mini_batch_y_pred)
        TP,FP,FN = (np.sum(i) for i in (TP,FP,FN))
        TP_l[c]+=TP
        FP_l[c]+=FP
        FN_l[c]+=FN

macro_f1(y_true,y_pred), np.mean([f1_score(TP_l[c],FP_l[c],FN_l[c]) for c in range(1,4) ])
# %%
