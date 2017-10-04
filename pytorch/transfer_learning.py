'''
reusing a previously trained network for a new task.  This is super useful and 
it turns out it works pretty darn well.  code taken from pytorch tutorial.
'''


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

plt.ion() # interactive mode
