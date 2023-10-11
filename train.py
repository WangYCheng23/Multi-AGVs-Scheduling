import json
import pprint
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR   
from tqdm import tqdm
from Environments.dispatch_agv.Dispatch_AGV_Env import *
from App.preprocessing import *
from Policies.Agent import *
from Policies.Utils import *


params = init_params(train=True)
add_env_info(params, "./env_info_data.json")
add_agent_info(params)
add_trainer_hypeparameters(params)

writer = SummaryWriter(params["PATH_TIME"]+"/")
device = params['DEVICE']
""" ENV SETUP """
env = DispatchAGVEnv(params=params)
num_agents = len(env.possible_agents)
num_observation = sum([
    env.observation_space(env.possible_agents[0]).spaces["buffer_rate"].shape[-1],
    env.observation_space(env.possible_agents[0]).spaces["next_dest"].shape[-1],
    # env.observation_space(env.possible_agents[0]).spaces["machine_status"].shape[-1],
    env.observation_space(env.possible_agents[0]).spaces["other_agent_destination"].shape[-1],
    env.observation_space(env.possible_agents[0]).spaces["self_agent_location"].shape[-1],
    env.observation_space(env.possible_agents[0]).spaces["self_agent_load"].shape[-1],
])
num_actions = env.action_space(env.possible_agents[0]).n

""" LEARNER SETUP """
agent = Agent(hidden_size=params["HIDDEN_SIZE"], num_observation=num_observation,num_actions=num_actions).to(device)
optimizer = optim.Adam(agent.parameters(), lr=params["LEARNING_RATE"], eps=params["EPSILON"])
scheduler = CosineAnnealingLR(optimizer, T_max=20)

""" ALGO LOGIC: EPISODE STORAGE"""
rb_obs = torch.zeros((params['STEPS_PER_EPOCH'], params['NUM_TRANSP_AGENTS'], num_observation)).to(device)
rb_actions = torch.zeros((params['STEPS_PER_EPOCH'], params['NUM_TRANSP_AGENTS'])).to(device)
rb_logprobs = torch.zeros((params['STEPS_PER_EPOCH'], params['NUM_TRANSP_AGENTS'])).to(device)
rb_rewards = torch.zeros((params['STEPS_PER_EPOCH'], params['NUM_TRANSP_AGENTS'])).to(device)
rb_terms = torch.zeros((params['STEPS_PER_EPOCH'], params['NUM_TRANSP_AGENTS'])).to(device)
rb_values = torch.zeros((params['STEPS_PER_EPOCH'], params['NUM_TRANSP_AGENTS'])).to(device)

""" TRAINING LOGIC """
# train for n number of episodes
for episode in range(params['MAX_EPOCH']):
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
        for step in tqdm(range(0, params['STEPS_PER_EPOCH']), desc="Collect Experience"):
            # rollover the observation
            obs = Convert_Observation(next_obs, device)
            # print(obs)

            # get action from the agent
            # actions = [np.random.randint(0, params["NUM_RESOURCES"]) for _ in range(6)]  # [5,6,7,6,5,7]
            actions, logprobs, _, values = agent.get_action_and_value(obs)
            # print(actions)
            
            # execute the environment and log data
            next_obs, rewards, terms = env.step(actions)
            
            # # add to episode storage
            rb_obs[step] = obs
            rb_rewards[step] = batchify(rewards, device)
            rb_terms[step] = batchify(terms, device)
            rb_actions[step] = actions
            rb_logprobs[step] = logprobs
            rb_values[step] = values.flatten()

            # compute episodic return
            total_episodic_return += rb_rewards[step].cpu().numpy()

            # if we reach termination or truncation, end
            if any([terms[a] for a in terms]): # or any([truncs[a] for a in truncs]):
                end_step = step
                break
    # env.export_log()

    # bootstrap value if not done
    with torch.no_grad():
        rb_advantages = torch.zeros_like(rb_rewards).to(device)
        for t in reversed(range(end_step)):
            delta = (
                rb_rewards[t]
                + params["GAMMA"] * rb_values[t + 1] * rb_terms[t + 1]
                - rb_values[t]
            )
            rb_advantages[t] = delta + params["GAMMA"] * params["GAMMA"] * rb_advantages[t + 1]
        rb_returns = rb_advantages + rb_values

    # convert our episodes to batch of individual transitions
    b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
    b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
    b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
    b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
    b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
    b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

    # Optimizing the policy and value network
    b_index = np.arange(len(b_obs))
    clip_fracs = []
    for repeat in tqdm(range(params["REPEAT_PER_COLLECT"]), desc="Optimize Policy"):
        # shuffle the indices we use to access the data
        np.random.shuffle(b_index)
        for start in range(0, len(b_obs), params["BATCH_SIZE"]):
            # select the indices we want to train on
            end = start + params["BATCH_SIZE"]
            batch_index = b_index[start:end]

            # 新策略
            _, newlogprob, entropy, value = agent.get_action_and_value(
                b_obs[batch_index], b_actions.long()[batch_index]
            )
            # 新旧策略比
            logratio = newlogprob - b_logprobs[batch_index]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_fracs += [
                    ((ratio - 1.0).abs() > params["CLIP_COEF"]).float().mean().item()
                ]

            # normalize advantaegs
            advantages = b_advantages[batch_index]
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            # Policy loss
            pg_loss1 = -b_advantages[batch_index] * ratio
            pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                ratio, 1 - params["CLIP_COEF"], 1 + params["CLIP_COEF"]
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            value = value.flatten()
            v_loss_unclipped = (value - b_returns[batch_index]) ** 2
            v_clipped = b_values[batch_index] + torch.clamp(
                value - b_values[batch_index],
                -params["CLIP_COEF"],
                params["CLIP_COEF"],
            )
            v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - params["ENT_COEF"] * entropy_loss + v_loss * params["VF_COEF"]
            # assert loss<0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    print("\n-------------------------------------------\n")
    print(f"Training episode {episode}")
    print(f"Episodic Return: {np.mean(total_episodic_return)}")
    print(f"Episode Length: {end_step}")
    print("")
    print(f"Value Loss: {v_loss.item()}")
    print(f"Policy Loss: {pg_loss.item()}")
    print(f"Old Approx KL: {old_approx_kl.item()}")
    print(f"Approx KL: {approx_kl.item()}")
    print(f"Clip Fraction: {np.mean(clip_fracs)}")
    print(f"Explained Variance: {explained_var.item()}")
    print("\n-------------------------------------------\n")

    writer.add_scalar("Training_data/Episodic Return",np.mean(total_episodic_return),episode)
    writer.add_scalar("Training_data/Value Loss",v_loss.item(),episode)
    writer.add_scalar("Training_data/Policy Loss",pg_loss.item(),episode)
    writer.add_scalar("Training_data/Old Approx KL",old_approx_kl.item(),episode)
    writer.add_scalar("Training_data/Approx KL",approx_kl.item(),episode)
    writer.add_scalar("Training_data/Clip Fraction",np.mean(clip_fracs),episode)
    writer.add_scalar("Training_data/Explained Variance",explained_var.item(),episode)

    if episode%params["EXPORT_FREQUENCY"] == 0:
        print("\n-------------------------------------------\n")
        print(f"Save model at epoch-{episode}")
        if not os.path.exists(params["MODEL_TIME"]):
            os.makedirs(params["MODEL_TIME"])
        # 保存模型
        torch.save(agent, params["MODEL_TIME"]+f'/agent-{episode}.pt')
        print("\n-------------------------------------------\n")
