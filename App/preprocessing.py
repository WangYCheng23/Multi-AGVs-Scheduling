import os
import argparse
import torch
import json
import numpy as np
from datetime import datetime


def map_id(map_dict, resource_list):
    for i, r in enumerate(resource_list) :
        map_dict[r['id']] = i
        r['id'] = i

def map_machinetype(machinetype_list):
    map_dict = {}
    map_idx = 0
    for i, mt in enumerate(machinetype_list) :
        if not map_dict.__contains__(mt):
            map_dict[mt] = map_idx
            machinetype_list[i] = map_dict[mt]
            map_idx += 1
        else:
            machinetype_list[i] = map_dict[mt]

def process_data(params, env_info_data):
    """
    
    process the data received from UE

    """
    if isinstance(env_info_data, str):
        # print(os.getcwd())
        with open(os.path.join(os.getcwd(), env_info_data), 'r') as f:
            data = json.load(f)
    else:
        with open(os.path.join(params['PATH_TIME'],'env_info_data.json'), 'w') as f:
            json.dump(env_info_data, f)
        data = env_info_data
    transports = []
    machines = []
    sources = []
    sinks = []

    for resource in data['resources_data']:
        if resource['type'] == 'transport':
            transports.append(resource)
        elif resource['type'] == 'machine':
            machines.append(resource)
        elif resource['type'] == 'source':
            sources.append(resource)
        elif resource['type'] == 'sink':
            sinks.append(resource)
        else:
            raise ValueError

    # sort resources
    sorted(transports, key=lambda x: x['id'])
    sorted(machines, key=lambda x: x['id'])
    sorted(sources, key=lambda x: x['id'])
    sorted(sinks, key=lambda x: x['id'])

    map_dict = {}
    map_id(map_dict, transports)
    map_id(map_dict, machines)
    map_id(map_dict, sources)
    map_id(map_dict, sinks)

    for s in sources:
        s['resp_machines'] = sorted([map_dict[s] for s in s['resp_machines']])

    for s in sinks:
        s['resp_machines'] = sorted([map_dict[s] for s in s['resp_machines']])
    # the number of resources
    num_transport = len(transports)
    num_sources = len(sources)
    num_machines = len(machines)
    num_sinks = len(sinks)
    num_all_resources = num_sources + num_machines + num_sinks

    # machine groups divide
    machine_group = [m['machine_type'] for m in machines]
    map_machinetype(machine_group)

    l_ = machines + sources + sinks + transports
    loc_norm = np.array([i["location"] for i in l_])/np.max([i["location"] for i in l_], axis=0)
    for i, ll in enumerate(l_):
        ll["location"] = list(loc_norm[i].round(3))
    speed_norm = np.array([i["speed"] for i in transports])/np.max([i["speed"] for i in transports])
    for i, tt in enumerate(transports):
        tt["speed"] = round(speed_norm[i],3)
    # the location between resources    
    l = machines + sources + sinks
    loc_matrix = np.zeros(shape=(num_all_resources, num_all_resources), dtype=np.float32)  
    for i in range(loc_matrix.shape[0]):
        for j in range(loc_matrix.shape[1]):
            dis = np.abs(l[i]['location'][0] - l[j]['location'][0]) + np.abs(l[i]['location'][1] - l[j]['location'][1])
            loc_matrix[i][j] = dis

    return transports,machines,sources,sinks,machine_group,loc_matrix,map_dict

def init_params(train=True):
    parser = argparse.ArgumentParser(description="construct your own factory layout")

    parser.add_argument('--DEVICE', default='cuda:1' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--PRINT_CONSOLE', default=False, help='Extended print out during running, particularly for debugging')
    parser.add_argument('--EXPORT_FREQUENCY', default=50, help='Number of steps between csv-export of log-files')
    parser.add_argument('--EXPORT_LOGS', default=True, help='Turn on/off export of log-files')
    # parser.add_argument('--NUM_ORDERS', default=100000, help='large number to not run out of orders')
    # parser.add_argument('--EPSILON', default=0.000001, help='Small number larger than zero used as "marginal" time step or to compare values')
    # parser.add_argument('--EXPONENTIAL_SMOOTHING', default=0.01)
    # parser.add_argument('--MAX_STEPS', default=5000)
    parser.add_argument('--SEED', default=523)
    if train:
        parser.add_argument('--PATH_TIME', default="log/train_log/" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        parser.add_argument('--PATH_TIME', default="log/eval_log/" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--MODEL_TIME', default="Model/" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    params = parser.parse_args()
    params = vars(params)

    if not os.path.exists(params['PATH_TIME']):
        os.makedirs(params['PATH_TIME'])

    return params

def add_env_info(params, env_info_data):
    transports,machines,sources,sinks,machine_group,loc_matrix,map_dict = process_data(params, env_info_data)
    # add parameters
    params.update({'NUM_TRANSP_AGENTS': len(transports)})
    params.update({'NUM_MACHINES': len(machines)})
    params.update({'NUM_SOURCES': len(sources)})
    params.update({'NUM_SINKS': len(sinks)})
    params.update({'NUM_RESOURCES': len(machines)+len(sources)+len(sinks)})
    params.update({'MAP_DICT':map_dict})

    params.update({'AGV_MAP_LIST':list(map_dict.keys())[:(len(transports))]})
    params.update({'MACHINE_MAP_LIST':list(map_dict.keys())[(len(transports)):(len(transports)+len(machines))]})
    params.update({'SOURCE_MAP_LIST':list(map_dict.keys())[-(len(sources)+len(sinks)):-(len(sinks))]})
    params.update({'SINK_MAP_LIST':list(map_dict.keys())[-(len(sinks)):]})
    params.update({'RESOURCE_MAP_LIST':list(map_dict.keys())[len(transports):]})
    # params.update({'NUM_PROD_VARIANTS': 1})
    # params.update({'NUM_PROD_STEPS': 1})
    # Transport parameters
    params.update({'TRANSP_LOCATION':[x['location'] for x in transports]})
    params.update({'TRANSP_SPEED': [x['speed'] for x in transports]})
    # params.update({'RESP_AREA_TRANSP': [[[True for _ in range(params['NUM_RESOURCES'])] for _ in range(params['NUM_RESOURCES'])] for _ in range(params['NUM_TRANSP_AGENTS'])]})
    # Source parameters
    params.update({'SOURCE_LOCATION':[x['location'] for x in sources]})
    params.update({'SOURCE_CAPACITIES': [x['capacity'] for x in sources]})
    # params.update({'RESP_AREA_SOURCE':[x['resp_machines'] for x in sources]})   # Orders for which machines are created in the specific source
    params.update({'MTOG': [x['order_generation_time'] for x in sources]})  # Mean Time Order Generation
    # params.update({'SOURCE_ORDER_GENERATION_TYPE': "MEAN_ARRIVAL_TIME"})  # Alternatives: ALWAYS_FILL_UP, MEAN_ARRIVAL_TIME
    # Machine parameters
    params.update({'MACHINE_LOCATION':[x['location'] for x in machines]})
    # params.update({'MACHINE_AGENT_TYPE': "FIFO"})  # Alternatives: FIFO -> Decision rule for selecting the next available order from the load port
    # params.update({'MACHINE_GROUPS': machine_group}) # Machines in the same group are able to perform the same process and are interchangeable
    params.update({'MACHINE_CAPACITIES': [x['capacity'] for x in machines]})  # Capacity for in and out machine buffers together
    # Sink parameters
    params.update({'SINK_LOCATION':[x['location'] for x in sinks]})
    # params.update({'RESP_AREA_SINK':[x['resp_machines'] for x in sinks]})

    params.update({'MIN_PROCESS_TIME': [x["process_time"][0] for x in machines]})
    params.update({'AVERAGE_PROCESS_TIME': [sum(x["process_time"])/len(x["process_time"]) for x in machines]})
    params.update({'MAX_PROCESS_TIME': [x["process_time"][1] for x in machines]})
    params.update({'CHANGEOVER_TIME': 0.0})  # Default: Not used
    # TO DO
    params.update({'MTBF': [1000.0] * params['NUM_MACHINES']})  # Unscheduled breakdowns
    params.update({'MTOL': [200.0] * params['NUM_MACHINES']})

    # Order parameters
    # params.update({'ORDER_DISTRIBUTION': [1.0 / params['NUM_MACHINES']] * params['NUM_MACHINES']})  # Probability which machine allocated, when orders are created
    # params.update({'VARIANT_DISTRIBUTION': [1.0 / params['NUM_PROD_VARIANTS']] * params['NUM_PROD_VARIANTS']})  # Probability which product variant, when orders are created

    # Handling time
    params.update({'TIME_TO_LOAD_MACHINE': [x['load_machine_time'] for x in transports]})
    params.update({'TIME_TO_UNLOAD_MACHINE': [x['unload_machine_time'] for x in transports]})
    params.update({'TIME_TO_LOAD_SOURCE': [x['load_source_time'] for x in transports]})
    params.update({'TIME_TO_UNLOAD_SOURCE': [x['unload_source_time'] for x in transports]})

    # Transport time
    params.update({'TRANSP_DISTANCE': [loc_matrix]*params['NUM_TRANSP_AGENTS']})
    params.update({'TRANSP_TIME': [params['TRANSP_DISTANCE'][i] / params['TRANSP_SPEED'][i] for i in range(params['NUM_TRANSP_AGENTS'])]})
    params.update({'MAX_TRANSP_TIME': np.array(params['TRANSP_TIME']).max()})

def add_agent_info(params):
    # In this setting the RL-agent (TRPO-Algorithm) is controlling the transport decision making
    params.update({'TRANSP_AGENT_TYPE': "PPO"})  # Alternativen: TRPO, FIFO, NJF, EMPTY
    # State Design
    # params.update({'TRANSP_AGENT_STATE': ['rel_buffer_fill_in_out', 'bin_machine_failure']})  # Alternatives: bin_buffer_fill, bin_machine_failure, bin_location, int_buffer_fill, rel_buffer_fill, rel_buffer_fill_in_out, order_waiting_time, order_waiting_time_normalized, distance_to_action, remaining_process_time, total_process_time
    # Reward Desgin
    # params.update({'TRANSP_AGENT_REWARD': "utilization"})  # Alternatives: valid_action, utilization, waiting_time_normalized, throughput, conwip, const_weighted, weighted_objectives
    # params.update({'TRANSP_AGENT_REWARD_SPARSE': ""})  # Alternatives: valid_action, utilization, waiting_time
    # params.update({'TRANSP_AGENT_REWARD_EPISODE_LIMIT': 0})  # Episode limit counter, default = 0
    # params.update({'TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE': "valid"})  # Alternatives: valid, entry, exit, time
    # params.update({'TRANSP_AGENT_REWARD_SUBSET_WEIGHTS': [1.0, 1.0]})  # Standard: [1.0, 1.0]  |  First: Const weight values for action to machine, Second: weight for action to sink
    # params.update({'TRANSP_AGENT_REWARD_OBJECTIVE_WEIGHTS': {'valid_action':2.0, 'utilization': 1.0, 'waiting_time': 1.0}})
    # params.update({'TRANSP_AGENT_REWARD_VALID_ACTION': 10})
    # params.update({'TRANSP_AGENT_REWARD_WAITING_ACTION': -5})
    # params.update({'TRANSP_AGENT_REWARD_INVALID_ACTION': -2})
    # params.update({'TRANSP_AGENT_MAX_INVALID_ACTIONS': 5})  # Number of invalid actions until forced action is choosen
    # params.update({'TRANSP_AGENT_WAITING_TIME_ACTION': 2})  # Waiting time of waiting time action
    # params.update({'TRANSP_AGENT_ACTION_MAPPING': 'direct'})  # Alternatives: direct, resource
    # params.update({'TRANSP_AGENT_WAITING_ACTION': False})  # Alternatives: True, False
    # params.update({'TRANSP_AGENT_EMPTY_ACTION': False})  # Alternatives: True, False
    # params.update({'TRANSP_AGENT_CONWIP_INV': 15})  # ConWIP inventory target if conwip reward is selected
    # params.update({'WAITING_TIME_THRESHOLD': 1000})  # Forced order transport if threshold reached

def add_trainer_hypeparameters(params):
    params.update({'GAMMA':0.99})
    params.update({'ENT_COEF':0.1})
    params.update({'VF_COEF':0.1})
    params.update({'CLIP_COEF':0.1})
    params.update({'HIDDEN_SIZE':64})
    params.update({'LEARNING_RATE':3e-4})
    params.update({'EPSILON':1e-5})

    params.update({'MAX_EPOCH':5000})
    params.update({'STEPS_PER_EPOCH':5000})
    params.update({'REPEAT_PER_COLLECT':10})
    params.update({'BATCH_SIZE':512})
    # params.update({'STEP_PER_COLLECT':500})
    # params.update({'EPISODE_PER_TEST':10})
    # params.update({'LOG_MODE':'train'})
    # params.update({'LEARNING_RATE':0.00005})
    # params.update({'REPLAY_BUFF_SIZE':200000})


if __name__ == "__main__":
    pass
