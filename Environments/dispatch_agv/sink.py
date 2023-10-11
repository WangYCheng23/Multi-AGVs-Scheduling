import simpy
from resources import Resource

class Sink(Resource):
    def __init__(self, env, id, parameters, resources, data_gen_utils, location, label, log):
        super().__init__(parameters, resources, data_gen_utils, location)
        self.env = env
        self.id = id
        self.label = label
        self.type = 'sink'
        self.buffer_in = []
        self.counter = 0
        self.log = log
    
    def storing(self):
        pass