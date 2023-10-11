import os, sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-1])
sys.path.append(root_path)
import json
import gymnasium
import simpy
import pandas as pd
import numpy as np
from gymnasium.utils import EzPickle, seeding

from pettingzoo import AECEnv, ParallelEnv
from machine import Machine
from source import Source
from sink import Sink
from agv import AGV
from utils import Data_Gen_Utils


class DispatchAGVEnv(ParallelEnv, EzPickle):
    def __init__(self, params):
        EzPickle.__init__(self)

        self.parameters = params

        # self.init_env = simpy.Environment()
        self.n_agvs = self.parameters["NUM_TRANSP_AGENTS"]
        self.n_sources = self.parameters["NUM_SOURCES"]
        self.n_machines = self.parameters["NUM_MACHINES"]
        self.n_sinks = self.parameters["NUM_SINKS"]
        self.n_resources = self.n_sources + self.n_machines + self.n_sinks

        self.agents = ["AGV_" + str(n) for n in range(self.n_agvs)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_agvs))))

        self.observation_spaces = self.observations()
        self.action_spaces = self.actions()    
    
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]
    
    def action_space(self, agent) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def observations(self):
        # ... {
        # ...   "agv_1": [[...]],
        # ...   "agv_2": [[...]],
        # ...   "agv_3": [[...]],
        # ...   "agv_4": [[...]],
        # ... }
        return dict(
            zip(
                self.agents,
                [
                    gymnasium.spaces.Dict(
                        {
                            # 1.buffer capacity rate
                            "buffer_rate":gymnasium.spaces.Box(0, 1, (1, (self.n_sources+self.n_machines)), np.float32),
                            # 2.order next destination
                            "next_dest":gymnasium.spaces.Box(-1, self.n_resources, (1, self.n_sources+self.n_machines), np.int8),
                            # 3.machine status break/unbreak
                            # "machine_status":gymnasium.spaces.MultiBinary(self.n_machines),
                            # 4.other agent destination (-1 means None)
                            "other_agent_destination":gymnasium.spaces.Box(-1, self.n_resources, (1, self.n_agvs-1), np.int8),
                            # 5.self agent location or in transporting (-1 means in transporting)
                            "self_agent_location":gymnasium.spaces.Box(-1, self.n_resources, (1, 1), np.int8),
                            # 6.self load/unload
                            "self_agent_load":gymnasium.spaces.MultiBinary(1)
                        }
                    )                    
                ]
                *self.n_agvs
            )
        )

    def actions(self):
        # ... {
        # ...   "agv_1": 1,
        # ...   "agv_2": 4,
        # ...   "agv_3": 7,
        # ...   "agv_4": 0, #不做动作
        # ... }
        return dict(
            zip(self.agents, [gymnasium.spaces.Discrete(self.n_resources)]*self.n_agvs)
        )

    def reset(self, seed=2022):
        "simpy environment reset and init"
        self.steps = 0
        self.log = dict()
        # col = ["action", "sim_time", "at_ID", "duration"]
        for i in range(self.n_machines):
            self.log.update({'machine'+'_'+str(i):list()})
        for i in range(self.n_sources):
            self.log.update({'source'+'_'+str(i):dict()})
        for i in range(self.n_sinks):
            self.log.update({'sink'+'_'+str(i):list()})
        for i in range(self.n_agvs):
            self.log.update({'agv'+'_'+str(i):list()}) 

        self.env = simpy.Environment()
        self.parameters.update({'stop_criteria':[self.env.event() for _ in range(self.parameters['NUM_TRANSP_AGENTS'])]})
        self.parameters.update({'step_criteria':[self.env.event() for _ in range(self.parameters['NUM_TRANSP_AGENTS'])]})
        self.parameters.update({'continue_criteria':[self.env.event() for _ in range(self.parameters['NUM_TRANSP_AGENTS'])]})

        # self.parameters.update({'agv_wait_action_queue':[]})
        # self.agv_wait_action_queue = []
        self.data_gen_utils = Data_Gen_Utils(seed, self.parameters)
        self.resources = self.init_resources()

        self.env.run(until=simpy.AllOf(self.env, self.parameters['step_criteria']))

        # self.env.run(until=simpy.AllOf(self.env, self.parameters['step_criteria']))
        # 计算s
        obs = dict(
            zip(
                self.agents,
                [self.resources['agvs'][i].calculate_state() for i in range(len(self.resources['agvs']))]
            )
        )
        return obs

    def step(self, action):

        # if self.steps == 1:
        #     agents = [a for a in self.resources['agvs']]
        #     self.env.run(until=simpy.AllOf(self.env, self.parameters['step_criteria']))
        # 给到单个agv action
        agents = [a for a in self.resources['agvs'] if a.wait_action==True]
        if len(agents)==0:
            raise AssertionError
        # print("-"*30 + str(len(agents)) + "-"*30)
        # 得到a
        for a in agents:
                a.next_destination = self.resources['all_resources'][action[a.id]]
                # a.make_sense.succeed()
                # a.make_sense = a.env.event()
                # 继续simpy仿真， 直到某个agv需要动作
                # if a.init:
                self.parameters['continue_criteria'][a.id].succeed()   # 应该继续啊？
                self.parameters['continue_criteria'][a.id] = self.env.event()
        # self.agv_wait_action_queue = []
        self.env.run(until=simpy.AllOf(self.env, self.parameters['step_criteria']))
        self.steps += 1
        terminal = dict(
            zip(
                self.agents,
                [False for i in range(len(self.resources['agvs']))] 
            )
        )
        if self.steps == self.parameters["STEPS_PER_EPOCH"]:
            terminal = dict(
                zip(
                    self.agents,
                    [True for _ in range(len(self.resources['agvs']))] 
                )
            )
           
        # if all(terminal):
        #     self.export_log()

        # 计算r
        rewards = dict(
            zip(
                self.agents,
                [self.resources['agvs'][i].calculate_reward() for i in range(len(self.resources['agvs']))] 
            )
        )
        # 计算s'
        next_obs = dict(
            zip(
                self.agents,
                [self.resources['agvs'][i].calculate_state() for i in range(len(self.resources['agvs']))]
            )
        )
        return next_obs, rewards, terminal

    def init_resources(self):
        re = dict()

        re.update({
            "machines":[Machine(
                env=self.env, 
                id=i, 
                capacity=self.parameters['MACHINE_CAPACITIES'][i], 
                parameters=self.parameters,
                resources=re, 
                data_gen_utils=self.data_gen_utils,
                location=self.parameters['MACHINE_LOCATION'][i],
                label=None,
                log = self.log['machine'+'_'+str(i)] 
            ) for i in range(self.n_machines)]
        })

        re.update({
            "sources":[Source(
                env=self.env,
                id=i,
                capacity=self.parameters['SOURCE_CAPACITIES'][i],
                parameters=self.parameters,
                resources=re,
                data_gen_utils=self.data_gen_utils,
                location=self.parameters['SOURCE_LOCATION'][i],
                label=None,
                log=self.log['source'+'_'+str(i)] 
            ) for i in range(self.n_sources)]
        })

        re.update({
            "sinks":[Sink(
                env=self.env,
                id=i,
                parameters=self.parameters,
                resources=re,
                data_gen_utils=self.data_gen_utils,
                location=self.parameters['SINK_LOCATION'][i],
                label=None,
                log = self.log['sink'+'_'+str(i)]
            ) for i in range(self.n_sinks)]
        })
        
        re.update({
            "agvs":[AGV(
                env=self.env,
                id=i,
                speed=self.parameters['TRANSP_SPEED'][i],
                parameters=self.parameters,
                resources=re,
                data_gen_utils=self.data_gen_utils,
                location=self.parameters['TRANSP_LOCATION'][i],
                label=None,
                log = self.log['agv'+'_'+str(i)]
            ) for i in range(self.n_agvs)]
        })

        re.update({
            "all_resources":re["machines"]+re["sources"]+re["sinks"]
        })

        return re

    def export_log(self):
        if self.parameters['EXPORT_LOGS']:
            log_path = self.parameters['PATH_TIME']+'/resources_log'
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            for key, value in self.log.items():
                if key.split('_')[0]=='source':
                    type_path = log_path+'/'+key.split('_')[0]+'s'
                    if not os.path.exists(type_path):
                        os.makedirs(type_path)
                    with open(type_path + f'/{key}.json', 'w') as f:
                        json.dump(value, f)
                else:
                    type_path = log_path+'/'+key.split('_')[0]+'s'
                    if not os.path.exists(type_path):
                        os.makedirs(type_path)
                    with open(type_path + f'/{key}.json', 'w') as f:
                        json.dump(value, f)
    