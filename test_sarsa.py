#!/usr/bin/env python3
import numpy as np
import gym
from sarsa import SarsaLambda, StateActionFeatureVectorWithTile
from mountain_car import MountainCarEnvWithStops as MountainCar

def test_sarsa_lamda():
    env = MountainCar()
    gamma = 1.

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45,.035,1]),
        # tile_width=np.array([.451,.0351,1]),
        axis=range(env.observation_space.low.ndim-1),
        # axis=range(env.observation_space.low.ndim),
    )

    # w = SarsaLambda(env, gamma, 0.8, 0.01, X, 1000)
    w = SarsaLambda(env, gamma, 0.8, 0.005, X, 1000)

    def greedy_policy(s,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s,done)
            s,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G

    Gs = [_eval() for _ in  range(100)]
    # _eval(True)

    assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'

if __name__ == "__main__":
    test_sarsa_lamda()
