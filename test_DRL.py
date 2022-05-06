import itertools
import os
import numpy as np
from sac import SoftActorCritic as SAC
from mountain_car import MountainCarEnvWithStops as MountainCar

if __name__ == '__main__':
    from parameters import RM
    seeds = range(1,1+3)
    gammas = [
        0.99,
        # 1.0,
    ]

    train_total_timesteps = int(1e5)
    eval_num_eps = 1
    eval_num_steps = 1000

    train_dir = "logs/train"
    eval_dir = "logs/eval"
    model_dir = "logs/model"
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(eval_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)

    # for observable_RM, rm_choice, gamma, seed in itertools.product(
    #     [True],
    #     [1],#range(len(RM)),
    #     gammas,
    #     seeds,
    # ):
    for observable_RM in [True]:
        for rm_choice in [1]:#range(len(RM)):
            for gamma in gammas:
                for seed in seeds:
                    algo_name = "SAC"
                    filename = f"{algo_name}_obs={observable_RM}_RM={rm_choice}_gamma={gamma}_seed={seed}"
                    train_path = os.path.join(train_dir, filename+".pkl")
                    eval_path = os.path.join(eval_dir, filename+".pkl")
                    model_path = os.path.join(model_dir, filename+".zip")
                    # if os.path.exists(eval_path): continue
                    print(filename)
                    trans_reward, stops = RM[rm_choice]
                    env = MountainCar(**RM[rm_choice],
                                      gamma=gamma,
                                      observable_RM=observable_RM,
                                      discrete_action=False,
                                      seed=seed)
                    model = SAC(env=env, path=train_path, total_timesteps=train_total_timesteps, gamma=gamma, seed=seed)
                    print("train")
                    # model.train()
                    # model.save(path=model_path)
                    model.load(path=model_path)
                    print("eval")
                    model.evaluate(path=eval_path, num_eps=eval_num_eps, num_steps=eval_num_steps, render=True)