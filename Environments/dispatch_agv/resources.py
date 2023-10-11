class Resource(object):
    def __init__(self, parameters, resources, data_gen_utils, location):
        # self.statistics = statistics
        self.parameters = parameters
        self.resources = resources
        # self.agents = agents
        self.location = location
        self.data_gen_utils = data_gen_utils