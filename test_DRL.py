import itertools
import os
import numpy as np
from sac import SoftActorCritic as SAC
from mountain_car import MountainCarEnvWithStops as MountainCar

if __name__ == '__main__':
    from parameters import RM
    seeds = [1,
             2,
             3,
             ]
    gammas = [
        0.99,
        1.0,
    ]
    rm_choices = [
        0,
        1,
    ]
    observable_RM_choices = [
        True,
        False,
    ]

    train_total_timesteps = int(250000)
    eval_num_eps = 1
    eval_num_steps = 1000

    train_dir = "logs/train"
    eval_dir = "logs/eval"
    model_dir = "logs/model"
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(eval_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)

    for observable_RM in observable_RM_choices:
        for rm_choice in rm_choices:
            for gamma in gammas:
                for seed in seeds:
                    algo_name = "SAC"
                    filename = f"{algo_name}_obs={observable_RM}_RM={rm_choice}_gamma={gamma}_seed={seed}"
                    train_path = os.path.join(train_dir, filename+".pkl")
                    eval_path = None #os.path.join(eval_dir, filename+".pkl")
                    model_path = os.path.join(model_dir, filename+".zip")
                    trans_reward, stops = RM[rm_choice]
                    env = MountainCar(**RM[rm_choice],
                                      gamma=gamma,
                                      observable_RM=observable_RM,
                                      discrete_action=False,
                                      seed=seed)
                    model = SAC(env=env, path=train_path, total_timesteps=train_total_timesteps, seed=seed)
                    print(filename)
                    # print("train")
                    # model.train()
                    # model.save(path=model_path)
                    model.load(path=model_path)
                    print("eval")
                    model.evaluate(path=eval_path, num_eps=eval_num_eps, num_steps=eval_num_steps, render=True)