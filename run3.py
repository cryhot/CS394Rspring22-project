#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil
import logging
import numpy as np
from typing import *
import json
import pickle
import gym
from gym.wrappers import Monitor

from utils import Args, RenderWrapper, GifWrapper
from rl.features import *
from rl.values import *
from rl.network import *
from rl.algo import *
from mountain_car import MountainCarEnvWithStops as MountainCar



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    logs_dir = "logs2"
    train_dir = os.path.join(logs_dir, "train")
    eval_dir = os.path.join(logs_dir, "eval")
    model_dir = os.path.join(logs_dir, "model")
    param_dir = os.path.join(logs_dir, "param")
    video_dir = os.path.join(logs_dir, "video")
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(eval_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)
    os.makedirs(param_dir,exist_ok=True)
    os.makedirs(video_dir,exist_ok=True)

    dry_run = False
    train_total_timesteps = int(1e5)
    eval_num_eps = 100
    eval_num_steps = 1

    # dry_run = True

    from parameters import set_seed

    env_Args = Args(
        observable_RM = [
            True,
            # False,
        ],
        gamma = [
            1.0,
            # 0.99,
        ],
        discrete_action = [True],
    )
    from parameters import RM
    # rm_choices = range(len(RM))
    rm_choices = [0]

    def feature_Q_functions(env, alpha=None) -> Generator[Tuple[ValueFunctionWithFeatureVector,str],None,None]:
        state_features = [
            TileFeatureVector.uniform_offsets( # MDP_state
                env.observation_space.low[:2],
                env.observation_space.high[:2],
                width=np.array([.45,.035]),
                num=10,
            ),
        ]
        state_indices = [
            [0,1],
        ]
        if env.observable_RM:
            state_features.extend([
                OneHotFeatureVector( # RM_state
                    round(env.observation_space.high[-1])+1,
                ),
            ])
            state_indices.extend([
                2,
            ])
        if env.discrete_action:
            action_feature = OneHotFeatureVector( # action features
                env.action_space.n,
            )
        else:
            raise NotImplementedError()
        
        yield ValueFunctionWithFeatureVector(
            X = ProductFeatureVector(
                ProductFeatureVector( # state features
                    *state_features, state_indices=state_indices,
                ),
                action_feature, # action features
            ),
            alpha = alpha,
        ), "Q=tiles"
    
    def network_Q_functions(env, alpha=None) -> Generator[Tuple[NNValueFunctionFromFeatureVector,str],None,None]:
        # if env.observable_RM:
        #     RM_features = [
        #         # (
        #         #     [LinearFeatureVector.identity(1)], # RM_state
        #         #     [2],
        #         #     ["RMenc=Linear"],
        #         # ),
        #         (
        #             [OneHotFeatureVector(round(env.observation_space.high[-1])+1)], # RM_state
        #             [2],
        #             ["RMenc=OneHot"],
        #         ),
        #     ]
        # else:
        #     RM_features = [
        #         (
        #             [],
        #             [],
        #             [],
        #         ),
        #     ]
        
        # for RM_state_features, RM_state_indices, RM_filename_part in RM_features:
        #     state_features = [
        #         *RM_state_features,
        #         LinearFeatureVector.identity(2), # MDP_state
        #     ]
        #     state_indices = [
        #         *RM_state_indices,
        #         [0,1],
        #     ]
        #     filename_part = [
        #         *RM_filename_part,
        #     ]

        if env.observable_RM:

            yield NNValueFunctionFromFeatureVector(
                SumFeatureVector(
                    SumFeatureVector( # state features
                        OneHotFeatureVector( # RM_state
                            round(env.observation_space.high[-1])+1,
                        ),
                        LinearFeatureVector.identity(2), # MDP_state
                        state_indices=(2, [0,1],),
                    ),
                    OneHotFeatureVector( # action features
                        env.action_space.n,
                    ),
                ),
                alpha = alpha,
            ), "Q=NN_RMenc=OneHot_Aenc=OneHot"
            
            # yield NNValueFunctionFromFeatureVector(
            #     SumFeatureVector(
            #         SumFeatureVector( # state features
            #             LinearFeatureVector.identity(1), # RM_state
            #             LinearFeatureVector.identity(2), # MDP_state
            #             state_indices=(2, [0,1],),
            #         ),
            #         OneHotFeatureVector( # action features
            #             env.action_space.n,
            #         ),
            #     ),
            #     alpha = alpha,
            # ), "Q=NN_RMenc=Linear_Aenc=OneHot"

            yield CompositeValueFunctionFromFeatureVector(
                SumFeatureVector(
                    SumFeatureVector( # state features
                        LinearFeatureVector.identity(1), # RM_state
                        state_indices=(2,),
                    ),
                    state_indices=(0,),
                ),
                lambda x: NNValueFunctionFromFeatureVector(
                    SumFeatureVector(
                        SumFeatureVector( # state features
                            LinearFeatureVector.identity(2), # MDP_state
                            state_indices=([0,1],),
                        ),
                        OneHotFeatureVector( # action features
                            env.action_space.n,
                        ),
                    ),
                    alpha = alpha,
                ),
            ), "Q=NN_RMenc=NNs_Aenc=OneHot"
        
        else:

            yield NNValueFunctionFromFeatureVector(
                SumFeatureVector(
                    LinearFeatureVector.identity(2).flat, # MDP_state
                    OneHotFeatureVector( # action features
                        env.action_space.n,
                    ),
                ),
                alpha = alpha,
            ), "Q=NN_Aenc=OneHot"

    methods = [
        ("SarsaLambda",
            SarsaLambda,
            Args(
                lam = [0.95],
                alpha = [0.1],
            ),
            (lambda *args, **kwargs: filter((lambda f: f[1] in ["Q=tiles"]),
                feature_Q_functions(*args, **kwargs),
            )),
        ),
        ("NStepSarsa",
            n_step_Sarsa,
            Args(
                n = [8],
                alpha = [0.05],
            ),
            (lambda *args, **kwargs: filter((lambda f: f[1] in ["Q=tiles"]),
                feature_Q_functions(*args, **kwargs),
            )),
        ),
        # ("NStepSarsa",
        #     n_step_Sarsa,
        #     Args(
        #         n = [8],
        #         alpha = [0.0001],
        #     ),
        #     (lambda *args, **kwargs: filter((lambda f: f[1] in ["Q=NN_RMenc=NNs_Aenc=OneHot", "Q=NN_Aenc=OneHot"]),
        #         network_Q_functions(*args, **kwargs),
        #     )),
        # ),
    ]

    train_seeds = range(1,1+3)
    r=0

    for (
        train_seed,
        (algo_name, algo, algo_Args, Q_functions),
        env_args,
        rm_choice,
    ) in itertools.product(
        train_seeds,
        methods,
        env_Args.product(),
        rm_choices,
    ):
        env = MountainCar(*env_args.args, **env_args.kwargs, **RM[rm_choice])
        for algo_args in algo_Args.product():
            for Q, Qfilename in Q_functions(env, alpha=algo_args.alpha): # this should be the inner loop
    
                params_json = dict(
                    algo=algo_name,
                    env_args=env_args,
                    RM=rm_choice,
                    algo_args=algo_args,
                    algo_Q=Qfilename,
                    seed=train_seed,
                    train_total_timesteps=train_total_timesteps,
                )

                filename = "_".join([
                    algo_name,
                    f"obs={env.observable_RM}_RM={rm_choice}_gamma={env.gamma}_A={['continuous','discrete'][env.discrete_action]}",
                    Qfilename,
                    *(f"{k}={v}" for k,v in algo_args.items()),
                    f"seed={train_seed}",
                ])
                train_path = os.path.join(train_dir, filename+".pkl")
                eval_path = os.path.join(eval_dir, filename+".pkl")
                if isinstance(Q, ValueFunctionWithFeatureVector):
                    model_path = os.path.join(model_dir, filename+".npy")
                else:
                    model_path = os.path.join(model_dir, filename+".NONE")
                param_path = os.path.join(param_dir, filename+".json")
                video_path = os.path.join(video_dir, filename+".gif")
                if os.path.exists(param_path):
                    with open(param_path, 'r') as f:
                        if params_json != json.load(f):
                            print(f">>> {filename}")
                            print(json.dumps(params_json, indent=2))
                            assert False, "Warning! param file not coherent."
                
                if os.path.exists(eval_path):
                    # with open(eval_path, 'rb') as f:
                    #     data = pickle.load(f)
                    #     # if len(data[0]) != 1000: continue # 1st episode length
                    #     if len(data) >= eval_num_steps: continue # episodes count
                    # continue # already evaluated
                    pass
                r+=1
                print(f">>> #{r} {filename}")
                if dry_run: continue
                

                if not os.path.exists(model_path) and os.path.exists(model_path.replace("logs2/","logs/")):
                    with open(param_path.replace("logs2/","logs/"), 'r') as f: data = json.load(f)
                    if params_json == data:
                        shutil.copy2(model_path.replace("logs2/","logs/"), model_path)
                        shutil.copy2(train_path.replace("logs2/","logs/"), train_path)

                if os.path.exists(eval_path):
                    pass
                elif os.path.exists(model_path):
                    print("LOAD")
                    if isinstance(Q, ValueFunctionWithFeatureVector):
                        Q.w = np.load(model_path)
                    else:
                        raise NotImplementedError()
                else:
                    print(f"TRAIN")
                    env.seed(train_seed)
                    set_seed(train_seed)
                    # env = RenderWrapper(env, fps=120)
                    data = run(
                        algo, env, Q, *algo_args.args, **algo_args.kwargs,
                        tot_iterations=train_total_timesteps,
                    )
                    with open(train_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(param_path, 'w') as f:
                    json.dump(params_json, f, indent=2)

                if isinstance(Q, ValueFunctionWithFeatureVector):
                    np.save(model_path, Q.w)
                        
                if not os.path.exists(eval_path):
                    print("EVAL")
                    data = []
                    for eval_seed in range(1,1+eval_num_eps):
                        env.seed(eval_seed)
                        set_seed(eval_seed)
                        data.extend(run(
                            evaluate, env, Q,
                            episodes=1,
                            ep_iterations=eval_num_steps,
                        ))
                    with open(eval_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print("REPLAY")
                with open(eval_path, 'rb') as f: data = pickle.load(f)
                eval_seed = 1
                env.seed(eval_seed)
                set_seed(eval_seed)
                env.reset()
                gifenv = GifWrapper(env, max_frame=500)
                for step in data[0][['s','r','done']]:
                    gifenv.step(0, replay=step)
                gifenv.save(filename=video_path, fps=120)

                print()
