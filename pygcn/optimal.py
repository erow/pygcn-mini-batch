from torch.utils import *
from torch import nn
import torch,math,os
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
#%%
mu,logvar = nn.Parameter(torch.randn(40)), nn.Parameter(torch.randn(40))
