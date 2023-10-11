import simpy
from resources import Resource

class Machine(Resource):
    def __init__(self, env, id, capacity, parameters, resources, data_gen_utils, location, label, log):
        super().__init__(parameters, resources, data_gen_utils, location)
        self.env = env
        self.id = id
        self.label = label
        self.type = 'machine'
        self.capacity = capacity    # 目前都指代出的容量
        self.counter = 0
        self.buffer_in = []
        self.buffer_out = []
        self.in_processing = False
        self.broken = False
        self.idle = env.event()
        self.process = self.env.process(self.processing())
        self.log = log
        self.log.append(["action", "sim_time", "at_ID", "duration"])
        # self.env.process(self.break_machine())

    def start_processing(self):
        self.idle = self.env.event()  # Reset idle event
        self.process = self.env.process(self.processing())

    def processing(self):
        while True:
            if len(self.buffer_in)==0:
                self.idle.succeed()
                break
            order = self.buffer_in.pop(0)
            # self.buffer_in.pop(index=0)
            self.in_processeing = True
            yield self.env.timeout(2.5)
            # while self.in_processeing:
            #     try:            
            #         yield self.env.timeout(2.5)
            #     except simpy.Interrupt:
            #         self.broken = True
            #         with self.resources['repairman'].request(priority=1) as req:
            #             yield req
            #             yield self.timeout()
            #         self.broken = False
            self.in_processeing = False
            self.buffer_out.append(order)
            order.processed.succeed()

    def break_machine(self):
        """Break the machine every now and then."""
        # while True:
        #     yield self.env.timeout()
        #     if not self.broken:
        #         if self.idle.triggered:
        #             self.broken = True
        #             yield self.env.timeout()
        #         else:
        #             self.process.interrupt()
        pass

# def other_jobs(env, repairman):
#     """    The repairman's other (unimportant) job.    """
#     while True:
#         done_in = 1000.0
#         while done_in:
#             # Retry the job until it is done. It's priority is lower than that of machine repairs.
#             with repairman.request(priority=2) as req:
#                 yield req
#                 try:
#                     start_time = env.now
#                     yield env.timeout(done_in)
#                     done_in = 0
#                 except simpy.Interrupt:
#                     done_in -= env.now - start_time
            