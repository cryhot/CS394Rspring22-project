#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from typing import *
import json
import pickle

from utils import Args, RenderWrapper, GifWrapper
from rl.features import *
from rl.values import *
from rl.network import *
from rl.algo import *
from mountain_car import MountainCarEnvWithStops as MountainCar


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    logs_dir = "logs"
    train_dir = os.path.join(logs_dir, "train")
    eval_dir = os.path.join(logs_dir, "eval")
    model_dir = os.path.join(logs_dir, "model")
    param_dir = os.path.join(logs_dir, "param")
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(eval_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)
    os.makedirs(param_dir,exist_ok=True)

    dry_run = False
    train_total_timesteps = int(1e5)
    eval_num_eps = 1
    eval_num_steps = 10000

    # dry_run = True

    from parameters import set_seed

    env_Args = Args(
        observable_RM = [
            True,
            False,
        ],
        gamma = [
            1.0,
            0.99,
        ],
        discrete_action = [True],
    )
    env_Args2 = Args(
        observable_RM = [
            True,
            False,
        ],
        gamma = [1.0],
        discrete_action = [True],
    )
    from parameters import RM

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
                lam = [
                    0.95,
                    0.9,
                    0.8,
                    0.6,
                ],
                alpha = [
                    0.2,
                    0.1,
                    0.05,
                ]
            ),
            (lambda *args, **kwargs: itertools.chain(
                feature_Q_functions(*args, **kwargs),
            )),
        ),
        ("NStepSarsa",
            n_step_Sarsa,
            Args(
                n = [
                    # 1,
                    # 2,
                    4,
                    8,
                ],
                alpha = [
                    0.2,
                    0.1,
                    0.05,
                ],
            ),
            (lambda *args, **kwargs: itertools.chain(
                feature_Q_functions(*args, **kwargs),
            )),
        ),
        ("NStepSarsa",
            n_step_Sarsa,
            Args(
                n = [
                    # 1,
                    # 2,
                    4,
                    8,
                ],
                alpha = [
                    0.01,
                    0.005,
                    0.001,
                    0.0005,
                    0.0001,
                ],
            ),
            (lambda *args, **kwargs: itertools.chain(
                network_Q_functions(*args, **kwargs),
            )),
        ),
    ]


    methods2 = [
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
        ("NStepSarsa",
            n_step_Sarsa,
            Args(
                n = [8],
                alpha = [0.0001],
            ),
            (lambda *args, **kwargs: filter((lambda f: f[1] in ["Q=NN_RMenc=NNs_Aenc=OneHot", "Q=NN_Aenc=OneHot"]),
                network_Q_functions(*args, **kwargs),
            )),
        ),
    ]

    seeds = range(1,1+3)
    r=0


    # for ( # first search (lot of parameters)
    #     seed,
    #     (algo_name, algo, algo_Args, Q_functions),
    #     env_args,
    #     rm_choice,
    # ) in itertools.product(
    #     seeds,
    #     methods,
    #     env_Args.product(),
    #     range(len(RM)),
    # ):
    #     env = MountainCar(*env_args.args, **env_args.kwargs, **RM[rm_choice])
    #     for algo_args in algo_Args.product():
    #         for Q, Qfilename in Q_functions(env, alpha=algo_args.alpha): # this should be the inner loop

    for (
        seed,
        (algo_name, algo, algo_Args, Q_functions),
        env_args,
        rm_choice,
    ) in itertools.product(
        # seeds,
        range(1,1+1000),
        methods2,
        env_Args2.product(),
        [0],
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
                    seed=seed,
                    train_total_timesteps=train_total_timesteps,
                )

                filename = "_".join([
                    algo_name,
                    f"obs={env.observable_RM}_RM={rm_choice}_gamma={env.gamma}_A={['continuous','discrete'][env.discrete_action]}",
                    Qfilename,
                    *(f"{k}={v}" for k,v in algo_args.items()),
                    f"seed={seed}",
                ])
                train_path = os.path.join(train_dir, filename+".pkl")
                eval_path = os.path.join(eval_dir, filename+".pkl")
                if isinstance(Q, ValueFunctionWithFeatureVector):
                    model_path = os.path.join(model_dir, filename+".npy")
                else:
                    model_path = os.path.join(model_dir, filename+".NONE")
                param_path = os.path.join(param_dir, filename+".json")
                if os.path.exists(param_path):
                    with open(param_path, 'r') as f:
                        if params_json != json.load(f):
                            print(f">>> {filename}")
                            print(json.dumps(params_json, indent=2))
                            assert False, "Warning! param file not coherent."
                
                if os.path.exists(eval_path):
                    with open(eval_path, 'rb') as f:
                        data = pickle.load(f)
                        if len(data[0]) != 1000: continue
                    # continue # already evaluated
                r+=1
                print(f">>> #{r} {filename}")
                if dry_run: continue
                

                if os.path.exists(model_path):
                    print("LOAD")
                    if isinstance(Q, ValueFunctionWithFeatureVector):
                        Q.w = np.load(model_path)
                    else:
                        raise NotImplementedError()
                else:
                    print(f"TRAIN")
                    env.seed(seed)
                    set_seed(seed)
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
                        
                

                print("EVAL")
                env.seed(seed)
                set_seed(seed)
                data = run(
                    evaluate, env, Q,
                    episodes=eval_num_eps,
                    ep_iterations=eval_num_steps,
                )
                with open(eval_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print()
