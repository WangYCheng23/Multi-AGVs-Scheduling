import os
import json
from this import d
import pandas as pd


def log2json(params):
    res = []
    file_path = params['PATH_TIME']+'/resources_log'
    name = os.listdir(file_path)
    for n in name:
        resource_name = n.split('.')[0]
        resource_type = resource_name.split('_')[0]
        resource_id = resource_name.split('_')[1]
        if resource_type == 'agv':
            agv_id = resource_id
            df2list = pd.read_csv(file_path+'/'+resource_name+'.csv').values.tolist()
            for x in df2list[1:]:           
                d = {'agvId':agv_id, 'action':x[0], 'sim_time':x[1], 'to_at':x[3], 'duration':x[4]}
                # res.append(json.dumps(d, ensure_ascii=False))
                res.append(d)
    # print(len(res))
    return {"data":res}


if __name__ == "__main__":
    pass