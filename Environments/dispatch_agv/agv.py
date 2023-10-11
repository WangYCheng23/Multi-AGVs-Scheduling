import numpy as np
import simpy
from resources import Resource

class AGV(Resource):
    def __init__(self, env, id, speed, parameters, resources, data_gen_utils, location, label, log):
        super().__init__(parameters, resources, data_gen_utils, location)
        self.env = env
        self.id = id
        self.speed = speed
        self.label = label
        self.type = 'agv'
        self.next_destination = None    # cls
        self.idle = env.event()
        # self.make_sense = env.event()   # 保证收到step给的action
        # self.init = False
        self.wait_action = False
        self.env.process(self.transporting())
        self.current_location = location
        self.current_order = None
        self.in_transporting = False
        self.steps_count = 0
        self.valid = False
        self.count_action = [0,0] # valid, invalid
        self.agv_log = log
        self.agv_log_key = ["agvId", "sim_time", "action", "to_at", "duration"]

    def calculate_reward(self):
        rew = 0
        # # 如果在运输中 那么给到-1作为动作是合理的
        # if self.in_transporting:
        #     return 0
        # # 1.基于buffer rate的评价
        # for i in self.resources['sources']:
        #     rew -= (round(len(i.buffer_out)/i.capacity, 4))
        # for j in self.resources['machines']:
        #     rew -= (round(len(j.buffer_out)/j.capacity, 4))
        # # 2.基于完成的订单数
        # a,b = 0, 0
        # for p in self.resources['sources']:
        #     a += p.counter
        # for q in self.resources['sinks']:
        #     b += q.counter
        # rew += round(b/a, 4)
        # # 3.基于机器利用率

        # 4.valid action
        if self.valid:
            rew += 5
        else:
            rew -= 1
        # 5.运输中。。
        
        return rew

    def calculate_state(self):
        obs = dict()
        # 1.buffer capacity rate
        cap_rate = []
        for i in self.resources['sources']:
            cap_rate.append(round(len(i.buffer_out)/i.capacity, 4))
        for j in self.resources['machines']:
            cap_rate.append(round(len(j.buffer_out)/j.capacity, 4))
        obs.update({"buffer_rate":cap_rate})
        # 2.order next destination
        order_nx_dest = []
        for i in self.resources['sources']:
            if i.buffer_out != []:
                order = i.buffer_out[0]
                order_nx_dest.append(order.procedures[order.actual_step].id) 
            else:
                order_nx_dest.append(-1)
        for j in self.resources['machines']:
            if j.buffer_out != []:
                order = j.buffer_out[0]
                order_nx_dest.append(order.procedures[order.actual_step].id) 
            else:
                order_nx_dest.append(-1)
        obs.update({"next_dest":order_nx_dest})
        # 3.machine status break/unbreak
        # mch_break = []
        # for i in self.resources['machines']:
        #     if i.broken:
        #         mch_break.append(-1.0)
        #     else:
        #         mch_break.append(1.0)
        # obs.update({"machine_status":mch_break})
        # 4.other agent destination
        other_ag_dest = []
        for i in self.resources['agvs']:
            if i == self:
                continue
            else:
                if i.next_destination == None:
                    other_ag_dest.append([-1.0, -1.0])
                else:
                    other_ag_dest.append(i.next_destination.location)
        obs.update({"other_agent_destination":other_ag_dest})
        # 5.self agent location or transported
        ag_loc_trans = []
        if not self.in_transporting:
            if isinstance(self.current_location, list):
                obs.update({"self_agent_location":self.current_location})
            else:
                obs.update({"self_agent_location":self.current_location.location})
        else:
            obs.update({"self_agent_location":[-1, -1]})
        # 6.self load/unload
        obs.update({"self_agent_load":[int(self.current_order!=None)]})
        return obs
    
    def load(self, resource):
        order = resource.buffer_out[0]
        resource.buffer_out = resource.buffer_out[1:]
        if resource.type == 'machine':
            order.order_log.append(["load_from_machine_"+str(resource.id), order.name, self.env.now, str(self.type) + '_' + str(self.id)])
            yield self.env.timeout(self.parameters["TIME_TO_LOAD_MACHINE"][resource.id])
        elif resource.type == 'source':
            order.order_log.append(["load_from_source_"+str(resource.id), order.name, self.env.now, str(self.type) + '_' + str(self.id)])
            yield self.env.timeout(self.parameters["TIME_TO_LOAD_SOURCE"][resource.id])
        return order

    def unload(self, order, resource):
        if resource.type == 'machine':
            order.order_log.append(["unload_to_machine_"+str(resource.id), order.name, self.env.now, str(self.type) + '_' + str(self.id)])
            yield self.env.timeout(self.parameters['TIME_TO_UNLOAD_MACHINE'][resource.id])
            resource.buffer_in.append(order) 
            resource.counter += 1
            order.current_location = resource
            order.transported.succeed()
            if resource.idle.triggered:
                resource.start_processing()
        elif resource.type == 'sink':
            order.order_log.append(["unload_to_sink_"+str(resource.id), order.name, self.env.now, str(self.type) + '_' + str(self.id)])
            yield self.env.timeout(0.2)
            resource.buffer_in.append(order)
            resource.counter += 1
            order.current_location = resource
            order.transported.succeed()
            
    def time_calculate(self, start, end):
        if not isinstance(start, list):
            start = start.location
        if not isinstance(end, list):
            end = end.location
        return (abs(start[0]-end[0])+abs(start[1]-end[1]))/self.speed

    def pick_destination(self):
        self.steps_count += 1

        # self.init=True
    #     self.parameters['reset_criteria'][self.id].succeed()
    #     yield self.env.timeout(np.random.rand())
        # self.wait_action = True
        self.parameters['step_criteria'][self.id].succeed()
        self.parameters['step_criteria'][self.id] = self.env.event()

        self.wait_action = True
        yield self.parameters['continue_criteria'][self.id]
        # self.parameters['step_criteria'][self.id] = self.env.event()
        # self.parameters['continue_criteria'][self.id] = self.env.event()
        # yield self.make_sense
        dest = self.next_destination
        self.wait_action = False

        return dest

    def transporting(self):
        while True:
            # 出发去哪里
            dest = yield self.env.process(self.pick_destination())
            # 计算前往的时间
            time2dest = self.time_calculate(start=self.current_location, end=dest)
            self.agv_log.append(
                dict(
                    zip(
                        self.agv_log_key,
                        [
                            self.parameters[f"{self.type.upper()}_MAP_LIST"][self.id], 
                            self.env.now, "transporting", 
                            self.parameters[f"{dest.type.upper()}_MAP_LIST"][dest.id], 
                            time2dest
                        ] 
                    )
                )      
            )
            self.in_transporting = True
            yield self.env.timeout(time2dest)
            self.in_transporting = False
            self.current_location = dest
            # 到达dest
            # 判断动作-奖励指标
            valid = False
            if self.current_order == None and hasattr(dest, 'buffer_out') and dest.buffer_out != []:    # 拉货
                order = yield self.env.process(self.load(dest))
                self.current_order = order
                valid = True
                self.count_action[0] += 1
                
            elif self.current_order != None and hasattr(dest, 'buffer_in') \
                    and self.current_order.procedures[self.current_order.actual_step]==dest:    # 放货
                yield self.env.process(self.unload(self.current_order, dest))
                self.current_order = None
                valid = True
                self.count_action[0] += 1
            
            else:
                # 驮着订单但不是该去的位置
                # 没有订单但没东西可装
                yield self.env.timeout(0.5)
                self.count_action[1] += 1

            self.valid = valid




