import torch
from tqdm import tqdm
from Policies.Agent import Agent
from Policies.Utils import *
from App.preprocessing import *
from App.postprocessing import *
from Environments.dispatch_agv.Dispatch_AGV_Env import *


model_path = "./Model/20221116_161613/agent-400.pt"
eval_agent = torch.load(model_path)
print(eval_agent)

params = init_params(train=False)
add_env_info(params, "./env_info_data.json")
add_agent_info(params)
add_trainer_hypeparameters(params)

device = params['DEVICE']
env = DispatchAGVEnv(params=params)

""" Evaluation LOGIC """
# train for n number of episodes
for episode in range(1):
    # collect an episode
    with torch.no_grad():
        # collect observations and convert to batch of torch tensors
        next_obs = env.reset(params["SEED"])
        # for key,value in next_obs.items():
        #     # pprint('{key}:{value}'.format(key = key, value = value))
        #     pprint.pprint(key)
        #     pprint.pprint(value, width=100)
        #     print('#'*80)
        # reset the episodic return
        total_episodic_return = 0
        # each episode has num_steps
        for step in range(0, 5000):
            # rollover the observation
            obs = Convert_Observation(next_obs, device)
            # print(obs)

            # get action from the agent
            actions = [np.random.randint(0, params["NUM_RESOURCES"]) for _ in range(params["NUM_TRANSP_AGENTS"])]  # [5,6,7,6,5,7]
            # actions, _, _, _ = eval_agent.get_action_and_value(obs)
            print(actions)
            
            # execute the environment and log data
            _, _, _ = env.step(actions)

    env.export_log()
    # post_data = log2json(params)