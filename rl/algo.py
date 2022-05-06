#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np




def SarsaLambda(
    env, # openai gym environment
    Q, # State-Action ValueFunctionWithFeatureVector
    *,
    lam:float, # decay rate
    alpha:float, # step size
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,epsilon=.0):
        nA = env.action_space.n
        Qs = [
            Q((s,a)) if not done else 0
            for a in range(nA)
        ]
        if np.random.rand() < epsilon:
            return np.random.randint(nA) # TODO: set seed if epsilon!=0
        else:
            return np.argmax(Qs)

    s0, r, done = env.reset(), 0., False
    a0 = epsilon_greedy_policy(s0, done)
    x0 = Q[s0,a0] if not done else np.zeros(Q.shape)
    Q0_old = 0
    z = np.zeros(Q.shape)  # eligibility trace vector

    while not done:
        s1,r,done,_ = env.step(a0)
        yield s0, a0, r, done
        a1 = epsilon_greedy_policy(s1, done)
        x1 = Q[s1,a1] if not done else np.zeros(Q.shape)
        Q0 = np.sum(Q.w*x0)
        Q1 = np.sum(Q.w*x1)
        delta = (r + env.gamma * Q1) - (Q0)  # Temporal-Difference error
        z = env.gamma*lam*z + (1-alpha*env.gamma*lam*np.sum(z*x0))*x0
        Q.w += alpha * (delta*z + (Q0-Q0_old)*(z-x0))

        Q0_old = Q1
        s0, a0, x0 = s1, a1, x1
    

def n_step_Sarsa(
    env, # openai gym environment
    Q, # State-Action ValueFunction
    *,
    n:int, # steps
    alpha:float, # step size
) -> np.array:
    """
    implement n-step semi gradient TD for estimating Q
    """

    def epsilon_greedy_policy(s,done,epsilon=.0):
        nA = env.action_space.n
        Qs = [
            Q((s,a)) if not done else 0
            for a in range(nA)
        ]
        if np.random.rand() < epsilon:
            return np.random.randint(nA) # TODO: set seed if epsilon!=0
        else:
            return np.argmax(Qs)
    
    gamma_i = np.power(env.gamma, np.arange(n))
    gamma_n = np.power(env.gamma, n)

    s0, cum_R, done = env.reset(), 0., False
    a0 = epsilon_greedy_policy(s0, done)
    buff = np.empty(n, dtype=[
        ('state',object), # tuple (s,a)
        ('r',float),
    ])
    buff['r'] = 0
    F = n # buff[F:]['S] are visited states
    B = n # buff[:B]['S] are yet to be updated
    # if not done: B+=1

    while B>0:
        if not done:
            s1, r, done, info = env.step(a0)
            yield s0, a0, r, done
            cum_R += r
            # if not done: B+=1
            a1 = epsilon_greedy_policy(s1, done)
        else:
            r = 0
            B -= 1

        buff = np.roll(buff,-1)
        buff[-1] = (s0,a0), r
        F = max(F-1, 0)

        if F <= 0: # if first state of the buffer is visited
            G = np.sum(gamma_i * buff['r'])
            # if not done: G += gamma_n * V(s1)
            if not done: G += gamma_n * Q((s1,a1))
            Q.update(buff['state'][0], G, alpha)

        s0, a0 = s1, a1

def evaluate(
    env, # openai gym environment
    Q, # State-Action ValueFunction
) -> np.array:
    """
    algorithm that just follows a Q-function based policy.
    """

    def policy(s,done):
        nA = env.action_space.n
        Qs = [
            Q((s,a)) if not done else 0
            for a in range(nA)
        ]
        return np.argmax(Qs)

    s0, cum_R, done = env.reset(), 0., False
    a0 = policy(s0, done)

    while not done:
        s1, r, done, info = env.step(a0)
        yield s0, a0, r, done
        a1 = policy(s1, done)

        s0, a0 = s1, a1