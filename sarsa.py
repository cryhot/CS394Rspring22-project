import itertools
import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
            state_low:np.array,
            state_high:np.array,
            num_actions:int,
            num_tilings:int,
            tile_width:np.array,
            axis=None,
        ):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        if axis is None: axis = range(state_low.ndim)
        tiles = np.zeros(shape=(num_actions, num_tilings), dtype=[
            ('low', state_low.dtype, state_low.shape),
            ('high', state_high.dtype, state_high.shape),
        ])
        # tiles['low'] = -np.linspace(0, tile_width, num_tilings, endpoint=False) # tiling starts
        for a in axis:
            # print(tiles['low'].shape)
            tiles['low'][...,a] = -np.linspace(0, tile_width[a], num_tilings, endpoint=False) # tiling starts
        for d, low, high, width in zip(itertools.count(), state_low, state_high, tile_width):
            tiles_low = np.arange(low, high+width, width)
            tiles = np.tile(tiles[...,np.newaxis], len(tiles_low))
            tiles['low'][...,d] += tiles_low
            tiles['high'][...,:-1,d] = tiles['low'][...,1:,d]
            tiles['high'][...,-1,d] = tiles['low'][...,-1,d]+width
        assert np.allclose(tiles['high'], tiles['low']+tile_width), "wrong tile width"
        assert np.allclose(tiles.shape[0], num_actions), "wrong number of actions"
        assert np.allclose(tiles.shape[1], num_tilings), "wrong number of tilings"
        assert np.allclose(tiles.shape[2:], np.ceil((state_high-state_low)/tile_width)+1), "wrong number of tiles"

        self.tiles = tiles

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # return self.tiles.shape
        return self.tiles.size

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        x = np.zeros(self.tiles.shape)
        if not done:
            x[a] = np.all(
                (self.tiles[a]['low'] <= s) & (s < self.tiles[a]['high']),
                axis = -1,
            )
            assert np.count_nonzero(x) == self.tiles.shape[1], "point not covered by the expected number of tiles"
        # return x
        return x.flat

def SarsaLambda(
    env, # openai gym environment
    X:StateActionFeatureVectorWithTile,
    *,
    w = None, # weight vector
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.sum(w*X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    if w is None: w = np.zeros(X.feature_vector_len())  # weight vector

    for episode in range(1,num_episode+1):
        frames = []
        print(f"episode {episode}/{num_episode}")
        s0, r, done = env.reset(), 0., False
        if episode == num_episode: frames.append(env.render(mode="rgb_array"))
        # s0 = s0.copy(); s0[-1]=0 # ignore RM states
        a0 = epsilon_greedy_policy(s0, done, w)
        x0 = X(s0,done,a0)
        Q0_old = 0
        z = np.zeros_like(w)  # eligibility trace vector

        while not done:
            s1,r,done,_ = env.step(a0)
            if episode == num_episode: frames.append(env.render(mode="rgb_array"))
            # s1 = s1.copy(); s1[-1]=0 # ignore RM states
            if episode == num_episode: env.render(mode="human")
            # env.render(fps=120)
            a1 = epsilon_greedy_policy(s1, done, w)
            x1 = X(s1,done,a1)
            Q0 = np.sum(w*x0)
            Q1 = np.sum(w*x1)
            delta = (r + gamma * Q1) - (Q0)  # Temporal-Difference error
            z = gamma*lam*z + (1-alpha*gamma*lam*np.sum(z*x0))*x0
            w += alpha * (delta*z + (Q0-Q0_old)*(z-x0))

            Q0_old = Q1
            s0, a0, x0 = s1, a1, x1
        
        if episode == num_episode:
            save_frames_as_gif(frames)

    return w


def n_step_Sarsa(
    env, # openai gym environment
    X:StateActionFeatureVectorWithTile,
    *,
    w = None, # weight vector
    gamma:float, # discount factor
    n:int, # steps
    alpha:float, # step size
    num_episode:int=1,
) -> np.array:
    """
    implement n-step semi gradient TD for estimating Q
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.sum(w*X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)
    
    if w is None: w = np.zeros(X.shape)  # weight vector
    gamma_i = np.power(gamma, np.arange(n))
    gamma_n = np.power(gamma, n)

    for episode in range(1,num_episode+1):
        print(f"episode {episode}/{num_episode}")
        s0, cum_R, done = env.reset(), 0., False
        a0 = epsilon_greedy_policy(s0, done, w)
        x0 = X(s0,done,a0)
        Q0_old = 0
        # env.render()
        buff = np.empty(n, dtype=[
            ('x',int, np.asarray(x0).shape),
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
                a1 = epsilon_greedy_policy(s1, done, w)
                x1 = X(s1,done,a1)
                Q0 = np.sum(w*x0)
                Q1 = np.sum(w*x1)
            else:
                r = 0
                B -= 1

            buff = np.roll(buff,-1)
            buff[-1] = x0, r
            F = max(F-1, 0)

            if F <= 0: # if first state of the buffer is visited
                G = np.sum(gamma_i * buff['r'])
                # if not done: G += gamma_n * V(s1)
                # V.update(alpha, G, buff['s'][0])
                if not done: G += gamma_n * Q1
                x_updt = buff['x'][0]
                Q_updt = np.sum(w*x_updt)
                w += alpha * (G - Q_updt) * x_updt

            s0, a0, x0 = s1, a1, x1




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

