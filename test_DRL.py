import gym
import numpy as np
from stable_baselines import PPO1, A2C, DQN, DDPG, HER, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec


from mountain_car import MountainCarEnvWithStops as MountainCar

env = MountainCar(discrete_action=False)


model = SAC(SACMlpPolicy, env, gamma=1., learning_rate=0.0003, buffer_size=50000, learning_starts=100, train_freq=1,
    batch_size=64, tau=0.005, ent_coef='auto', target_update_interval=1, gradient_steps=1, target_entropy='auto',
    action_noise=None, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True,
    policy_kwargs=None, full_tensorboard_log=False, seed=1, n_cpu_tf_sess=None)
model.learn(total_timesteps=int(1e5), log_interval=10)
model.save("sac_mountaincar")
del model # remove to demonstrate saving and loading
model = SAC.load("sac_mountaincar", env=env)

# model_class = DDPG
# model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy="future",
#                                                 verbose=1)
# model.learn(total_timesteps=int(1e5))
# model.save("her_ddpg_mountaincar")
# del model # remove to demonstrate saving and loading
# model = HER.load("her_ddpg_mountaincar", env=env)

# n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
# model = DDPG(DDPGMlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,
#              memory_limit=int(5e4), buffer_size=int(5e4))#, random_exploration=0.1)
# # model = DDPG(DDPGMlpPolicy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50, nb_rollout_steps=100,
# #              nb_eval_steps=100, param_noise=None, action_noise=None, normalize_observations=False,
# #              tau=0.001, batch_size=128, param_noise_adaption_interval=50, normalize_returns=False, enable_popart=False,
# #              observation_range=(-5.0, 5.0), critic_l2_reg=0.0, return_range=(-inf, inf), actor_lr=0.0001,
# #              critic_lr=0.001, clip_norm=None, reward_scale=1.0, render=False, render_eval=False, memory_limit=5000,
# #              buffer_size=5000, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True,
# #              policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
# model.learn(total_timesteps=int(1e5))
# model.save("ddpg_mountaincar")
# del model # remove to demonstrate saving and loading
# model = DDPG.load("ddpg_mountaincar")

# model = PPO1("MlpPolicy", env, gamma=0.99, timesteps_per_actorbatch=64, clip_param=0.2, entcoeff=0.01,
#              optim_epochs=4, optim_stepsize=0.001, optim_batchsize=32, lam=0.95, adam_epsilon=1e-03,
#              schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
#              policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
# model.learn(total_timesteps=int(1e5))
# print("learned model")
# model.save("ppo_mountaincar")
# print("saved model")
# del model  # delete trained model to demonstrate loading
# model = PPO1.load("ppo_mountaincar.zip")
# print("loaded model")

# REWARD FCN: SMALLER PUNISHMENT AND LARGER REWARD? #


# model = A2C(MlpPolicy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
#             learning_rate=0.0007, alpha=0.99, epsilon=1e-05, lr_schedule='constant',
#             verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
#             full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)
# model.learn(total_timesteps=int(1e5))
# print("learned model")
# model.save("a2c_mountaincar")
# print("saved model")
# del model  # delete trained model to demonstrate loading
# model = A2C.load("a2c_mountaincar.zip")
# print("loaded model")

# model = DQN(DQNMlpPolicy, env, gamma=0.99, learning_rate=0.0005, buffer_size=5000, exploration_fraction=0.1,
#             exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32,
#             double_q=True, learning_starts=1000, target_network_update_freq=100, prioritized_replay=True,
#             prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
#             prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
#             _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None)
# model.learn(total_timesteps=int(1e5))
# print("learned model")
# model.save("dqn_mountaincar")
# print("saved model")
# del model  # delete trained model to demonstrate loading
# model = DQN.load("dqn_mountaincar.zip")
# print("loaded model")

# Evaluate the agent
# print("evaluating model")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(mean_reward)
# print(std_reward)
# Enjoy trained agent
obs = env.reset()
print("evaluating model - 2")
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(fps=120)
    if done:
        break