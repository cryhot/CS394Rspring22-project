import gym
import pickle
import numpy as np
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, path="record.pkl", verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.s_buffer = []
        self.a_buffer = []
        self.ep_buffer = []
        self.r_buffer = []
        self.done_buffer = []
        self.path = path

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # print(self.locals.keys())
        # if self.locals["step"] > 0:
        #     if self.locals["num_episodes"] > 0:
        #         print("ep: "+str(self.locals["num_episodes"])+" || step: "+str(self.locals["step"])+" || reward: "+str(self.locals["reward"])+" || done: "+str(self.locals["done"]))
        # else:
        #     print("ep: ? || step: "+str(self.locals["step"])+" obs: "+str(self.locals["obs"])+" || reward: "+str(self.locals["reward"])+" || done: "+str(self.locals["done"]))

        self.ep_buffer.append(self.locals["num_episodes"] if self.locals["step"] > 0 else 0)
        self.s_buffer.append(self.locals["obs"])
        self.a_buffer.append(self.locals["action"])
        self.r_buffer.append(self.locals["reward"])
        self.done_buffer.append(self.locals["done"])
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        s_num = 3 if self.training_env.observable_RM else 2
        data = np.zeros(shape=len(self.ep_buffer), dtype=[
            ('episode', int),
            ('s', float, (s_num,)),
            ('a', int),
            ('r', float),
            ('done', bool),
        ], )
        data['episode'] = np.array(self.ep_buffer).ravel()
        data['s'] = np.array(self.s_buffer)
        data['a'] = np.array(self.a_buffer).ravel()
        data['r'] = np.array(self.r_buffer).ravel()
        data['done'] = np.array(self.done_buffer).ravel()

        with open(self.path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

class SoftActorCritic:
    def __init__(self, env, path, total_timesteps=int(1e5), gamma=.99, seed=1):
        self.total_timesteps = total_timesteps
        self.env = env
        self.model = SAC(SACMlpPolicy, env, gamma=gamma, learning_rate=0.0003, buffer_size=10000, learning_starts=100, train_freq=1,
            batch_size=64, tau=0.005, ent_coef='auto', target_update_interval=1, gradient_steps=1, target_entropy='auto',
            action_noise=None, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True,
            policy_kwargs=None, full_tensorboard_log=False, seed=seed, n_cpu_tf_sess=None)
        self.callback = CustomCallback(path=path)

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps, log_interval=10, callback = self.callback)

    def evaluate(self, num_eps, num_steps, path = None, render=False):
        s_buffer = []
        a_buffer = []
        ep_buffer = []
        r_buffer = []
        done_buffer = []
        for ep in range(num_eps):
            obs = self.env.reset()
            for t in range(num_steps):
                action, _states = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                if render:
                    self.env.render(fps=120)
                if path is not None:
                    ep_buffer.append(ep)
                    s_buffer.append(obs)
                    a_buffer.append(action)
                    r_buffer.append(reward)
                    done_buffer.append(done)
                if done:
                    break

        if path is not None:
            s_num = 3 if self.env.observable_RM else 2
            data = np.zeros(shape=len(ep_buffer), dtype=[
                ('episode', int),
                ('s', float, (s_num,)),
                ('a', int),
                ('r', float),
                ('done', bool),
            ], )
            data['episode'] = np.array(ep_buffer).ravel()
            data['s'] = np.array(s_buffer)
            data['a'] = np.array(a_buffer).ravel()
            data['r'] = np.array(r_buffer).ravel()
            data['done'] = np.array(done_buffer).ravel()
            with open(path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pass

    def save(self, path="sac_mountaincar"):
        self.model.save(path)

    def load(self, path="sac_mountaincar"):
        self.model = SAC.load(path, env=self.env)
