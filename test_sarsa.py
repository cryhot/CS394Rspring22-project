#!/usr/bin/env python3
import os
import numpy as np
import gym
from gym.wrappers import Monitor
from sarsa import StateActionFeatureVectorWithTile
from sarsa import SarsaLambda, n_step_Sarsa
from mountain_car import MountainCarEnvWithStops as MountainCar

def test_sarsa_lamda():
    gamma = 1.

    env = MountainCar()
    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45,.035,1]),
        # tile_width=np.array([.451,.0351,1]),
        axis=range(env.observation_space.low.ndim-1),
        # observable_RM=False,
    )

    # env = gym.make("MountainCar-v0")
    # X = StateActionFeatureVectorWithTile(
    #     env.observation_space.low,
    #     env.observation_space.high,
    #     env.action_space.n,
    #     num_tilings=10,
    #     tile_width=np.array([.45,.035]),
    #     # tile_width=np.array([.451,.0351]),
    # )

    w = None
    # if os.path.exists("weight_vector.npy"): w = np.load("weight_vector.npy")
    # w = SarsaLambda(env, X, gamma, lam=0.8, alpha=0.01, w=w, num_episode=1000)

    # w = SarsaLambda(env, X, gamma=gamma, lam=0.8, alpha=0.005, w=w, num_episode=100)
    w = n_step_Sarsa(env, X, gamma=gamma, n=10, alpha=0.005, w=w, num_episode=100)

    np.save("weight_vector.npy", w)

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
