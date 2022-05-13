#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import *
import itertools
import logging
import numpy as np
import termcolor



def run(algo, env, *args,
    tot_iterations=np.infty,
    ep_iterations=np.infty,
    episodes=np.infty,
    **kwargs,
):
    "Run an algo"
    tot_i = 0
    logs = []
    for e in itertools.count():
        if e >= episodes: return logs
        logging.debug(
            f"episode {e+1}"
            + (f"/{episodes}" if episodes<np.infty else "")
            + f" (tot_iter={tot_i}"
            + (f"/{tot_iterations}" if tot_iterations<np.infty else "")
            + ")"
        )
        logs_ep = []
        for ep_i,(s,a,r,done) in enumerate(algo(env, *args, **kwargs)):
            if tot_i >= tot_iterations:
                episodes = -1
                break
            if ep_i >= ep_iterations: break
            logs_ep.append((e,s,a,r,done))
            tot_i += 1
        logs_ep = np.array(logs_ep, dtype=[
            ('episode', int),
            ('s', env.observation_space.dtype, env.observation_space.shape),
            ('a', env.action_space.dtype, env.action_space.shape),
            ('r', float),
            ('done', bool),
        ])
        logs.append(logs_ep)



def learn_RM(alphabet,
    sample:Tuple['pos_sample','neg_sample'],
    pos_runs=[], neg_runs=[], # the infered DFA must comply with these, but we do enlarge the sample only when necessary.
    *,
    start_N=1, stop_N=np.inf, # formla size
) -> 'dfa':
    "Learns a Reward Machine efficiently. Modifies the sample inplace."
    from automata_learning_utils.pysat.sat_data_file import read_RPNI_samples, extract_samples_from_traces
    from automata_learning_utils.pysat.sat_data_file import sat_data_file
    from automata_learning_utils.pysat.problem import Problem
    pos_sample, neg_sample = sample
    N = start_N
    logging.debug(termcolor.colored(f">>> trying to solve DFA with {N} states.", color='blue', attrs=['bold']))
    while True:
        if N > stop_N: break
        problem = Problem(N, alphabet)
        # print(len(pos_sample),len(neg_sample))
        # for p in pos_sample: print('+',''.join(str(l) for l in p))
        # for n in neg_sample: print('-',''.join(str(l) for l in n))
        problem.add_positive_traces(pos_sample)
        problem.add_negative_traces(neg_sample)
        wcnf = problem.build_cnf()
        success = problem.solve()
        if not success:
            N+=1
            logging.debug(termcolor.colored(f">>> trying to solve DFA with {N} states.", color='blue', attrs=['bold']))
            continue
        dfa = problem.get_automaton()
        # if any(x==y for x in sample[0] for y in sample[1]): print("HHHHHHH")
        if update_sample(dfa, sample, pos_runs, neg_runs):
            logging.debug(termcolor.colored(f"... trying again.", color='blue', attrs=['bold']))
            continue
        logging.debug(termcolor.colored(f">>> found!", color='blue', attrs=['bold']))
        return dfa
    return None

def update_sample(dfa,
    sample:Tuple['pos_sample','neg_sample'],
    pos_runs=[], neg_runs=[],
):
    "If the sample is updated, returns False. Else, returns True. Modifies the sample inplace."
    pos_sample, neg_sample = sample
    for sub_run in itertools.chain(pos_runs, neg_runs): # we suppose that prefixes are negative runs
        for i in range(len(sub_run)):
            if sub_run[:i] in dfa:
                neg_sample.append(sub_run[:i])
                # logging.debug(termcolor.colored(f">>> bim 1!", color='red', attrs=['bold']))
                return True
    for pos_run in pos_runs:
        if pos_run not in dfa:
            pos_sample.append(pos_run)
            # logging.debug(termcolor.colored(f">>> bim 2!", color='red', attrs=['bold']))
            return True
    for neg_run in neg_runs:
        if neg_run in dfa:
            neg_sample.append(neg_run)
            # logging.debug(termcolor.colored(f">>> bim 3!", color='red', attrs=['bold']))
            return True
    return False




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