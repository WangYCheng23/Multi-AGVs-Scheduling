import numpy as np
import torch

def Convert_Observation(dic, device):
    """Convert observation to tensor"""
    t = []
    for _, d in dic.items():
        buffer_rate = np.array(d["buffer_rate"])
        next_dest = np.array(d["next_dest"])
        # machine_status = np.array(d["machine_status"])
        other_agent_destination = np.array(d["other_agent_destination"])
        self_agent_location = np.array(d["self_agent_location"])
        self_agent_load = np.array(d["self_agent_load"])

        other_agent_destination = np.array([np.sqrt(i[0]**2 + i[1]**2).round(4) for i in other_agent_destination])
        self_agent_location = np.array([np.sqrt(self_agent_location[0]**2 + self_agent_location[1]**2).round(4)])
        # t.append(np.concatenate((buffer_rate,next_dest,machine_status,other_agent_destination,self_agent_location,self_agent_load)))
        t.append(np.concatenate((buffer_rate,next_dest,other_agent_destination,self_agent_location,self_agent_load)))
    return torch.tensor(np.array(t), dtype=torch.float32).to(device)    

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)
    return x
    
