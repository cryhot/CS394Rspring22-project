#!/usr/bin/env python3
import os
import numpy as np
import gym
from gym.wrappers import Monitor
from rl.features import *
from rl.values import *
from rl.network import *
from sarsa import SarsaLambda, n_step_Sarsa
from mountain_car import MountainCarEnvWithStops as MountainCar

def test_sarsa_lamda():
    gamma = 1. # TODO: move in env

    # env = MountainCar(observable_RM=True)
    # X = ProductFeatureVector(
    #     TileFeatureVector.uniform_offsets( # state features
    #         env.observation_space.low,
    #         env.observation_space.high,
    #         width=np.array([.45,.035,1]), # TODO: split MDP and RM states
    #         num=10,
    #     ),
    #     OneHotFeatureVector( # action features
    #         env.action_space.n,
    #     ),
    # )
    # Q = ValueFunctionWithFeatureVector(X)

    # env = MountainCar(observable_RM=True)
    # X = ProductFeatureVector(
    #     ProductFeatureVector( # state features
    #         TileFeatureVector.uniform_offsets( # MDP_state
    #             env.observation_space.low[:-1],
    #             env.observation_space.high[:-1],
    #             width=np.array([.45,.035]),
    #             num=10,
    #         ),
    #         OneHotFeatureVector( # RM_state
    #             round(env.observation_space.high[-1])+1,
    #         ),
    #         state_indices=([0,1], 2,),
    #     ),
    #     OneHotFeatureVector( # action features
    #         env.action_space.n,
    #     ),
    # )
    # Q = ValueFunctionWithFeatureVector(X)


    # env = MountainCar(observable_RM=True)
    # Q = NNValueFunctionFromFeatureVector(
    #     SumFeatureVector(
    #         SumFeatureVector( # state features
    #             LinearFeatureVector.identity(1), # RM_state
    #             # OneHotFeatureVector( # RM_state
    #             #     round(env.observation_space.high[-1])+1,
    #             # ),
    #             LinearFeatureVector.identity(2), # MDP_state
    #             state_indices=(2, [0,1],),
    #         ),
    #         OneHotFeatureVector( # action features
    #             env.action_space.n,
    #         ),
    #     )
    # )

    
    env = MountainCar(observable_RM=True)
    Q = CompositeValueFunctionFromFeatureVector(
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
            )
        )
    )



    # env = MountainCar()
    # X = StateActionFeatureVectorWithTile(
    #     env.observation_space.low,
    #     env.observation_space.high,
    #     env.action_space.n,
    #     num_tilings=10,
    #     tile_width=np.array([.45,.035,1]),
    #     # tile_width=np.array([.451,.0351,1]),
    #     axis=range(env.observation_space.low.ndim-1),
    # )

    # env = gym.make("MountainCar-v0")
    # X = StateActionFeatureVectorWithTile(
    #     env.observation_space.low,
    #     env.observation_space.high,
    #     env.action_space.n,
    #     num_tilings=10,
    #     tile_width=np.array([.45,.035]),
    #     # tile_width=np.array([.451,.0351]),
    # )

    
    # Q = ValueFunctionWithFeatureVector(X)
    # if os.path.exists("weight_vector.npy"): w = np.load("weight_vector.npy")
    # SarsaLambda(env, Q, gamma=gamma, lam=0.8, alpha=0.01, episodes=1000)
    # SarsaLambda(env, Q, gamma=gamma, lam=0.8, alpha=0.1, episodes=1000)

    # SarsaLambda(env, Q, gamma=gamma, lam=0.8, alpha=0.005, episodes=1000)
    # n_step_Sarsa(env, Q, gamma=gamma, n=10, alpha=0.005, episodes=1000)
    n_step_Sarsa(env, Q, gamma=gamma, n=10, alpha=3e-4, episodes=1000)

    # np.save("weight_vector.npy", w)

    # def greedy_policy(s,done):
    #     Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
    #     return np.argmax(Q)

    # def _eval(render=False):
    #     s, done = env.reset(), False
    #     if render: env.render()

    #     G = 0.
    #     while not done:
    #         a = greedy_policy(s,done)
    #         s,r,done,_ = env.step(a)
    #         if render: env.render()

    #         G += r
    #     return G

    # Gs = [_eval() for _ in  range(100)]
    # # _eval(True)

    # assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'

if __name__ == "__main__":
    test_sarsa_lamda()
