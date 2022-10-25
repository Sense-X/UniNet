import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

from timm.models import create_model
from uninet import *

input_size = 224

model = create_model('UniNetB1')
model.eval()

flops = FlopCountAnalysis(model, torch.rand(1, 3, input_size, input_size))
print(flop_count_table(flops, max_depth=2))
