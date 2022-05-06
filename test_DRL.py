import os
import numpy as np
from sac import SoftActorCritic as SAC
from mountain_car import MountainCarEnvWithStops as MountainCar

if __name__ == '__main__':
    trans_reward = {0: -1,
                    1: -0.1}
    RM = {0:[# (state_center, state_width, reward)
        ([0.525, 0.035], [0.15, np.infty], 0),
        ([-0.5, 0.], [0.2, 0.02], 0)],
        1:[# (state_center, state_width, reward)
        ([0.525, 0.035], [0.15, np.infty], 10),
        ([-0.5, 0.], [0.2, 0.02], 100)]}
    seeds = [1,2,3]
    gammas = [0.99]#, 1.0

    train_total_timesteps = int(1e5)
    eval_num_eps = 1
    eval_num_steps = 1000

    train_dir = "logs/train/"
    eval_dir = "logs/eval/"
    model_dir = "logs/model/"
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(eval_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)

    for observable_RM in [True]:
        for rm_choice in [1]:#range(1):
            for gamma in gammas:
                for seed in seeds:
                    print("SAC_obs="+str(observable_RM)+"_RM="+\
                          str(rm_choice)+"_gamma="+ str(gamma)+"_seed="+str(seed))
                    env = MountainCar(trans_reward=trans_reward[rm_choice],
                                      stops=RM[rm_choice],
                                      gamma=gamma,
                                      observable_RM=observable_RM,
                                      discrete_action=False,
                                      seed=seed)
                    if observable_RM:
                        filename = "SAC_obs="+str(observable_RM)+"_RM="+\
                                     str(rm_choice)+"_gamma="+ str(gamma)+"_seed="+str(seed)
                        train_path = train_dir+filename+".pkl"
                        eval_path = eval_dir+filename+".pkl"
                        model_path = model_dir+filename+".zip"
                    else:
                        train_path = train_dir+"SAC_obs="+str(observable_RM)+"_RM="+\
                                     str(rm_choice)+"_gamma="+str(gamma)+"_seed="+str(seed)+".pickle"
                        eval_path = eval_dir+"SAC_obs="+str(observable_RM)+"_RM="+\
                                    str(rm_choice)+"_gamma="+str(gamma)+"_seed="+str(seed)+".pickle"
                        model_path = model_dir+"SAC_obs="+str(observable_RM)+"_RM="+\
                                     str(rm_choice)+"_gamma="+str(gamma)+"_seed="+str(seed)+".zip"
                    model = SAC(env=env, path=train_path, total_timesteps=train_total_timesteps, gamma=gamma, seed=seed)
                    print("train")
                    # model.train()
                    # model.save(path=model_path)
                    model.load(path=model_path)
                    print("eval")
                    model.evaluate(path=eval_path, num_eps=eval_num_eps, num_steps=eval_num_steps, render=True)