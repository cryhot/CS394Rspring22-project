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
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
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

    w = np.zeros(X.feature_vector_len())  # weight vector

    for episode in range(num_episode):
        print(f"episode {episode}/{num_episode}")
        s0, r, done = env.reset(), 0., False
        # s0 = s0.copy(); s0[-1]=0 # ignore RM states
        a0 = epsilon_greedy_policy(s0, done, w)
        x0 = X(s0,done,a0)
        Q0_old = 0
        z = np.zeros_like(w)  # eligibility trace vector

        while not done:
            s1,r,done,_ = env.step(a0)
            # s1 = s1.copy(); s1[-1]=0 # ignore RM states
            # if episode > 300: env.render()
            env.render()
            a1 = epsilon_greedy_policy(s1, done, w)
            x1 = X(s1,done,a1)
            Q0 = np.sum(w*x0)
            Q1 = np.sum(w*x1)
            delta = (r + gamma * Q1) - (Q0)  # Temporal-Difference error
            z = gamma*lam*z + (1-alpha*gamma*lam*np.sum(z*x0))*x0
            w += alpha * (delta*z + (Q0-Q0_old)*(z-x0))

            Q0_old = Q1
            s0, a0, x0 = s1, a1, x1

    return w
