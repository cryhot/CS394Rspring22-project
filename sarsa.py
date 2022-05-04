import numpy as np
from typing import *


def SarsaLambda(
    env, # openai gym environment
    Q, # State-Action ValueFunctionWithFeatureVector
    *,
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    episodes:int=1,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    if isinstance(episodes, int): episodes = range(1, episodes+1)

    def epsilon_greedy_policy(s,done,epsilon=.0):
        nA = env.action_space.n
        Qs = [
            Q((s,a)) if not done else 0
            for a in range(nA)
        ]
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Qs)

    for episode in episodes:
        frames = []
        print(f"episode {episode}/{episodes[-1]}")
        s0, r, done = env.reset(), 0., False
        if episode == episodes[-1]: frames.append(env.render(mode="rgb_array"))
        # s0 = s0.copy(); s0[-1]=0 # ignore RM states
        a0 = epsilon_greedy_policy(s0, done)
        x0 = Q[s0,a0] if not done else np.zeros(Q.shape)
        Q0_old = 0
        z = np.zeros(Q.shape)  # eligibility trace vector

        while not done:
            s1,r,done,_ = env.step(a0)
            if episode == episodes[-1]: frames.append(env.render(mode="rgb_array"))
            # s1 = s1.copy(); s1[-1]=0 # ignore RM states
            if episode == episodes[-1]: env.render(mode="human")
            if episode > 10: env.render(fps=120)
            # env.render(fps=120)
            a1 = epsilon_greedy_policy(s1, done)
            x1 = Q[s1,a1] if not done else np.zeros(Q.shape)
            Q0 = np.sum(Q.w*x0)
            Q1 = np.sum(Q.w*x1)
            delta = (r + gamma * Q1) - (Q0)  # Temporal-Difference error
            z = gamma*lam*z + (1-alpha*gamma*lam*np.sum(z*x0))*x0
            Q.w += alpha * (delta*z + (Q0-Q0_old)*(z-x0))

            Q0_old = Q1
            s0, a0, x0 = s1, a1, x1
        
        if episode == episodes[-1]:
            save_frames_as_gif(frames)

    # return w


def n_step_Sarsa(
    env, # openai gym environment
    Q, # State-Action ValueFunction
    *,
    gamma:float, # discount factor
    n:int, # steps
    alpha:float, # step size
    episodes:int=1,
) -> np.array:
    """
    implement n-step semi gradient TD for estimating Q
    """
    if isinstance(episodes, int): episodes = range(1, episodes+1)

    def epsilon_greedy_policy(s,done,epsilon=.0):
        nA = env.action_space.n
        Qs = [
            Q((s,a)) if not done else 0
            for a in range(nA)
        ]
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Qs)
    
    gamma_i = np.power(gamma, np.arange(n))
    gamma_n = np.power(gamma, n)

    for episode in episodes:
        print(f"episode {episode}/{episodes[-1]}")
        s0, cum_R, done = env.reset(), 0., False
        a0 = epsilon_greedy_policy(s0, done)
        # env.render()
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
                env.render(fps=120)
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




def save_frames_as_gif(frames, filename='./gym_animation.gif', episode_num=None, text_color=(0,0,0), fps=60):
    # code inspired by https://stackoverflow.com/a/65970345
    import os
    import imageio
    import numpy as np
    from PIL import Image
    import PIL.ImageDraw as ImageDraw
    import matplotlib.pyplot as plt  
    ims = []
    for frame in frames:
        im = Image.fromarray(frame)
        drawer = ImageDraw.Draw(im)
        if episode_num is not None:
            raise NotImplemented("printing episode_num is not working.")
            # drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)
        ims.append(im)
    imageio.mimwrite(filename, frames, fps=fps)

