import simpy
from resources import Resource

class Order(Resource):
    def __init__(self, env, id, name, procedures, parameters, resources, data_gen_utils, log):
        Resource.__init__(self, parameters, resources, data_gen_utils, None)
        self.env = env
        self.id = id
        self.name = name
        self.procedures = procedures
        self.sop = -1
        self.eop = -1
        self.finished = False
        self.actual_step = 0
        self.current_location = None
        self.transported = self.env.event()
        self.processed = self.env.event()
        self.order_log = log
        self.order_log.append(["action", "order_name", "sim_time", "resource_ID"])
        
    def set_sop(self):
        self.sop = self.env.now
        self.order_log.append(["start_processing", self.name, round(self.sop, 5), self.current_location.type+'_'+str(self.current_location.id)])

    def set_eop(self):
        self.eop = self.env.now
        self.order_log.append(["end_processing", self.name, round(self.eop, 5), self.current_location.type+'_'+str(self.current_location.id)])

    def order_processing(self):
        while True:
            # 工序+1
            self.actual_step += 1
            assert self.actual_step <= len(self.procedures)-1
            # if self.actual_step == len(self.procedures)-1:
            #     self.finished = True
            #     break
            
            # 等待被运输
            yield self.transported
            self.transported = self.env.event()

            # 如果被运到sink 就没必要加工
            if self.procedures[self.actual_step].type == "sink":
                self.finished = True
                break

            if self.procedures[self.actual_step].type == "machine":
                # 等待被加工
                yield self.processed
                self.processed = self.env.event()
            
        self.set_eop()
        self.current_location = None