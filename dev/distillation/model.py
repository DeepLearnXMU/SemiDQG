from torch.distributions import Categorical
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch


class T5ForConditionalGenerationKD(T5ForConditionalGeneration):
    ...
