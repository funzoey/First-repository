from tkinter import Y
import numpy as np
import torch
from builder_copy import backward2forward

a = torch.arange(0,5)
x = a != 0

c = torch.tensor([[1,2,3,4,5,6,7,6,9],[5,1,2,6,4,2,3,9,0]])
b, m = backward2forward(c, 9)

print(b, )