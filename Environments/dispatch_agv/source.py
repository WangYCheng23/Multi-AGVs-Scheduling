import numpy as np
import resource
import simpy
from resources import Resource
from order import Order

class Source(Resource):
    """
    上料机
    """
    def __init__(self, env, id, capacity, parameters, resources, data_gen_utils, location, label, log):
        Resource.__init__(self, parameters=parameters, resources=resources, data_gen_utils=data_gen_utils, location=location)
        # print("Source %s created" % id)
        self.env = env
        self.id = id
        self.label = label
        self.type = "source"
        self.idle = env.event()
        # self.resp_area = resp_area
        self.capacity = capacity
        self.buffer_out = []
        self.counter = 0
        self.env.process(self.order_creating())  # Process started at creation
        self.log = log
    
    def put_buffer_out(self, order):
        self.buffer_out.append(order)

    def order_creating(self):
        while True:
            # 防止订单一直产生 
            # 太多后直接trunc
            # 给负奖励
            if len(self.buffer_out) >= self.capacity:
                self.idle.succeed()
                break
            yield self.env.timeout(self.parameters["MTOG"][self.id])
            procedures = self.data_gen_utils.create_order_procedures(
                resources=self.resources, 
                at_resource=self
            )
            new_order_name = str(self.id) + '_' + str(self.counter)
            self.log.update({new_order_name:[]})
            order = Order(
                env=self.env, 
                id=self.counter, 
                name = new_order_name,
                procedures=procedures, 
                parameters=self.parameters, 
                resources=self.resources,
                data_gen_utils=self.data_gen_utils,
                log = self.log[new_order_name]
            ) 
            self.counter += 1
             
            order.current_location = self
            order.set_sop()
            self.put_buffer_out(order)
            self.env.process(order.order_processing())