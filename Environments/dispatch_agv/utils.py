import numpy as np
import torch
import random

class Data_Gen_Utils:
    def __init__(self, seed, parameters):
        self.seed(seed)
        self.parameters = parameters
    
    def seed(self, seed):
        torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
        torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
        torch.backends.cudnn.deterministic = True  # cudnn
        np.random.seed(seed)  # numpy
        random.seed(seed)  # random and transforms

    def random_location(self):
        return np.random.choice(self.parameters["NUM_RESOURCES"])

    def create_order_procedures(self, resources, at_resource):
        procedures = []
        # source choice
        procedures.append(at_resource)
        # machine choice
        procedures.append(resources['machines'][np.random.choice(self.parameters["NUM_MACHINES"])])
        # sink choice
        procedures.append(resources['sinks'][np.random.choice(self.parameters["NUM_SINKS"])])
        return procedures             