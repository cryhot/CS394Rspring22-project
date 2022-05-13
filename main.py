#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse


# == helper functions ======================================================== #
def create_parser(parser=None, name=None, **kwargs) -> argparse.ArgumentParser:
    """Return either the existing parser, either a fresh parser or subparser.
    If `help` is not explicitely given, it will be set to `description` when creating a subparser.
    """
    if parser is None:
        kwargs.pop('help', None)
        parser = argparse.ArgumentParser(**kwargs)
    elif isinstance(parser, argparse._SubParsersAction):
        if name is None: raise TypeError("must provide name when creating subparser")
        if 'help' not in kwargs:
            if 'description' in kwargs: kwargs['help'] = kwargs['description']
        parser = parser.add_parser(name, **kwargs)
    assert isinstance(parser, argparse.ArgumentParser)
    return parser

# == parsers and mains ======================================================= #

def print_help(args:argparse.Namespace):
    "Print parser help and exit."
    args.parser.print_help(file=sys.stderr)
    exit(2)

# -- env --------------------------------------------------------------------- #

def create_parser_env(parser=None):
    """:returns: parser"""
    import sys, argparse
    if parser is None:
        parser = argparse.ArgumentParser(
            description="""
                Create a Multi-task MountainCar environment.
            """,
        )
    parser.set_defaults(parser=parser)
    parser.set_defaults(parse_env=parse_env)

    group_env = parser.add_argument_group(title="environment parameters")

    group_env.add_argument('--discrete-actions',
        dest='discrete_action',
        type=int, choices=[0,1], default=1,
        help = "If true (default), the agent uses discrete actions, if false, it uses continuous actions.",
    )

    group_env.add_argument('--observable-RM',
        dest='observable_RM',
        type=int, choices=[0,1,2], default=0,
        help = "Makes the Reward Machine observable by the agent (0: no, 1: yes (given), 2: infer RM)",
    )
    
    group_env.add_argument('--RM', metavar="ID",
        type=int, default='0',
        choices=range(2),
        help = "Preconfigured Reward machine and Stops: 0 is (R=-1, Ra=0, Rb=0), 1 is (R=-0.1, Ra=10, Rb=100)",
    )

    group_env.add_argument('--gamma', metavar="FLOAT",
        type=float, default="1",
        help = "discount factor",
    )

    return parser
def parse_env(args:argparse.Namespace):
    from mountain_car import MountainCarEnvWithStops as MountainCar
    from parameters import RM
    # args.observable_RM = bool(args.observable_RM)
    args.discrete_action = bool(args.discrete_action)
    if 'RM' in args:
        RM_args = RM[args.RM]
    env = MountainCar(
        gamma=args.gamma,
        observable_RM=args.observable_RM,
        discrete_action=args.discrete_action,
        **RM_args,
    )
    return env

# -- ValueFunction ----------------------------------------------------------- #

def create_parser_TileCoding(parser=None):
    parser = create_parser(parser,
        name='TileCoding',
        description="""
            Q function approximator with Tile Coding.
        """,
        epilog="""
            TileCoding fixed parameters are
            tile_width = [.45,.035],
            num_tilings = 10.
        """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(parse_Q=parse_TileCoding)

    parser.add_argument('--save-model', metavar="PATH",
        dest='path_model_save',
        help="Save the weight vector as a numpy file (.npy)",
    )
    parser.add_argument('--load-model', metavar="PATH",
        dest='path_model_load',
        help="Load the weight vector and skip training",
    )

    return parser
def parse_TileCoding(args:argparse.Namespace, env, alpha):
    import numpy as np
    from rl.values import ValueFunctionWithFeatureVector
    from rl.features import (
        TileFeatureVector,
        OneHotFeatureVector,
        ProductFeatureVector,
    )
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
                len(env.RM_learned.states),
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
    
    Q = ValueFunctionWithFeatureVector(
        X = ProductFeatureVector(
            ProductFeatureVector( # state features
                *state_features, state_indices=state_indices,
            ),
            action_feature, # action features
        ),
        alpha = alpha,
    )
    return Q

def create_parser_Network(parser=None):
    parser = create_parser(parser,
        name='Network',
        description="""
            Q function approximator with Neural Network.
        """,
        epilog="""
            Neural Network fixed parameters are
            hidden_layers = 2
            neurons_per_hidden_layer = 32
        """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(parse_Q=parse_Network)

    group_NN = parser.add_argument_group(title="Neural Network parameters")

    group_NN.add_argument('--RMenc', #metavar="ENCODING",
        choices=['OneHot', 'Linear', 'NNs'], required=True,
        help = "NNs: one NN per RM state; otherwise, encode RM state as input neuron(s) (1 or one-hot).",
    )

    # parser.add_argument('--save-model', metavar="PATH",
    #     dest='path_model_save',
    #     help="Save the weight vector as a numpy file (.npy)",
    # )
    # parser.add_argument('--load-model', metavar="PATH",
    #     dest='path_model_load',
    #     help="Load the weight vector and skip training",
    # )

    return parser
def parse_Network(args:argparse.Namespace, env, alpha):
    import numpy as np
    from rl.network import NNValueFunctionFromFeatureVector
    from rl.values import CompositeValueFunctionFromFeatureVector
    from rl.features import (
        LinearFeatureVector,
        OneHotFeatureVector,
        SumFeatureVector,
    )

    if env.observable_RM:
        if args.RMenc == "OneHot":
            Q = NNValueFunctionFromFeatureVector(
                SumFeatureVector(
                    SumFeatureVector( # state features
                        OneHotFeatureVector( # RM_state
                            len(env.RM_learned.states),
                        ),
                        LinearFeatureVector.identity(2), # MDP_state
                        state_indices=(2, [0,1],),
                    ),
                    OneHotFeatureVector( # action features
                        env.action_space.n,
                    ),
                ),
                alpha = alpha,
            )
        elif args.RMenc == "Linear":
            Q = NNValueFunctionFromFeatureVector(
                SumFeatureVector(
                    SumFeatureVector( # state features
                        LinearFeatureVector.identity(1), # RM_state
                        LinearFeatureVector.identity(2), # MDP_state
                        state_indices=(2, [0,1],),
                    ),
                    OneHotFeatureVector( # action features
                        env.action_space.n,
                    ),
                ),
                alpha = alpha,
            )
        elif args.RMenc == "NNs":
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
                    ),
                    alpha = alpha,
                ),
            )
        else:
            raise NotImplementedError(f"RMenc={args.RMenc}")
    else:
        Q = NNValueFunctionFromFeatureVector(
            SumFeatureVector(
                LinearFeatureVector.identity(2).flat, # MDP_state
                OneHotFeatureVector( # action features
                    env.action_space.n,
                ),
            ),
            alpha = alpha,
        )
    return Q

# -- algos ------------------------------------------------------------------- #

def create_parser_NStepSarsa(parser=None):
    parser = create_parser(parser,
        name='NStepSarsa',
        description="""
            Episodic n-step Semi-gradient Sarsa.
        """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(main=run_NStepSarsa)

    group_algo = parser.add_argument_group(title="algorithm parameters")

    group_algo.add_argument('--n', metavar="INT",
        type=int, required=True,
        help = "steps",
    )
    group_algo.add_argument('--alpha', metavar="FLOAT",
        type=float, required=True,
        help = "step size",
    )

    subparsers_Q = parser.add_subparsers(metavar="Q",
        title="value function approximators",
        # required=True, # not working
        help="Q-function approximator to use",
    )
    create_parser_TileCoding(subparsers_Q)
    create_parser_Network(subparsers_Q)

    return parser
def run_NStepSarsa(args:argparse.Namespace):
    if not hasattr(args, 'parse_Q'):
        args.parser.error('the value function approximator Q is required')
    env = args.parse_env(args)

    import numpy as np
    import pickle
    import logging
    from rl.algo import n_step_Sarsa, evaluate
    from rl.algo import run
    from rl.algo import learn_RM, update_sample
    from utils import RenderWrapper
    from parameters import set_seed

    Q = args.parse_Q(args, env, args.alpha)
    logging.basicConfig(level=logging.DEBUG)
    rend_env = RenderWrapper(env, fps=120)

    if hasattr(args, 'path_model_load') and args.path_model_load is not None:
        logging.info("LOADING TRAINED MODEL")
        Q.w = np.load(args.path_model_load)
    else:
        logging.info("TRAINING")
        if hasattr(args, 'seed'):
            env.seed(args.seed)
            set_seed(args.seed)
        data = []
        if env.observable_RM==2: # use RM_learned
            all_pos_runs, all_neg_runs = [], []
            sample = ([], [])
            env.RM_learned = learn_RM(env.RM_learned.alphabet, sample, pos_runs=all_pos_runs, neg_runs=all_neg_runs) # "empty" RM
            env.RM_learned.export_as_visualization_dot(
                output_file="learned.dot",
                keep_states=True, keep_alphabet=True,
                group_separator=r';',
            )
            Q = args.parse_Q(args, env, args.alpha) # reset Q
        while True:
            remaining_iterations = args.train_tot_steps - sum(len(ep) for ep in data)
            if remaining_iterations <= 0: break
            d = run(
                n_step_Sarsa, (rend_env if args.train_render else env), Q,
                n=args.n, alpha=args.alpha,
                tot_iterations=args.train_tot_steps,
                # episodes=args.train_num_eps,
                # ep_iterations=args.train_num_steps,
            )
            data.extend(d)
            if env.observable_RM==2: # use RM_learned
                pos_runs, neg_runs = [], []
                d_run = [env.lbl_of(s['s']) for s in d[0]]
                d_run.append(env.lbl) # fix last element that got truncated
                (pos_runs if d[0][-1]['done'] else neg_runs).append(d_run)
                all_pos_runs.extend(pos_runs); all_neg_runs.extend(neg_runs)
                if update_sample(env.RM_learned, sample, pos_runs=pos_runs, neg_runs=neg_runs): # check if new counterexamples
                    env.RM_learned = learn_RM(env.RM_learned.alphabet, sample, pos_runs=all_pos_runs, neg_runs=all_neg_runs, start_N=len(env.RM_learned.states)) # relearn RM
                    env.RM_learned.export_as_visualization_dot(
                        output_file="learned.dot",
                        keep_states=True, keep_alphabet=True,
                        group_separator=r';',
                    )
                    Q = args.parse_Q(args, env, args.alpha) # reset Q
        if args.path_train_save is not None:
            with open(args.path_train_save, 'wb') as f: pickle.dump(data, f)
    if hasattr(args, 'path_model_save') and args.path_model_save is not None:
        np.save(args.path_model_save, Q.w)
    
    logging.info("EVALUATION")
    if hasattr(args, 'seed'):
        env.seed(args.seed)
        set_seed(args.seed)
    data = run(
        evaluate, (rend_env if args.eval_render else env), Q,
        # tot_iterations=args.eval_tot_steps,
        episodes=args.eval_num_eps,
        ep_iterations=args.eval_num_steps,
    )
    if args.path_eval_save is not None:
        with open(args.path_eval_save, 'wb') as f: pickle.dump(data, f)

def create_parser_SarsaLambda(parser=None):
    parser = create_parser(parser,
        name='SarsaLambda',
        description="""
            True Online Sarsa(lambda).
        """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(main=run_SarsaLambda)

    group_algo = parser.add_argument_group(title="algorithm parameters")

    group_algo.add_argument('--lambda', metavar="FLOAT",
        dest='lam', type=float, required=True,
        help = "decay rate",
    )
    group_algo.add_argument('--alpha', metavar="FLOAT",
        type=float, required=True,
        help = "step size",
    )

    subparsers_Q = parser.add_subparsers(metavar="Q",
        title="value function approximators",
        # required=True, # not working
        # help="Q-function approximator to use",
    )
    create_parser_TileCoding(subparsers_Q)

    return parser
def run_SarsaLambda(args:argparse.Namespace):
    if not hasattr(args, 'parse_Q'):
        args.parser.error('the value function approximator Q is required')
    env = args.parse_env(args)

    import numpy as np
    import pickle
    import logging
    from rl.algo import SarsaLambda, evaluate
    from rl.algo import run
    from rl.algo import learn_RM, update_sample
    from utils import RenderWrapper
    from parameters import set_seed

    Q = args.parse_Q(args, env, args.alpha)
    logging.basicConfig(level=logging.DEBUG)
    rend_env = RenderWrapper(env, fps=120)

    if hasattr(args, 'path_model_load') and args.path_model_load is not None:
        logging.info("LOADING TRAINED MODEL")
        Q.w = np.load(args.path_model_load)
    else:
        logging.info("TRAINING")
        if hasattr(args, 'seed'):
            env.seed(args.seed)
            set_seed(args.seed)
        data = []
        if env.observable_RM==2: # use RM_learned
            all_pos_runs, all_neg_runs = [], []
            sample = ([], [])
            env.RM_learned = learn_RM(env.RM_learned.alphabet, sample, pos_runs=all_pos_runs, neg_runs=all_neg_runs) # "empty" RM
            env.RM_learned.export_as_visualization_dot(
                output_file="learned.dot",
                keep_states=True, keep_alphabet=True,
                group_separator=r';',
            )
            Q = args.parse_Q(args, env, args.alpha) # reset Q
        while True:
            remaining_iterations = args.train_tot_steps - sum(len(ep) for ep in data)
            if remaining_iterations <= 0: break
            d = run(
                SarsaLambda, (rend_env if args.train_render else env), Q,
                lam=args.lam, alpha=args.alpha,
                tot_iterations=remaining_iterations,
                episodes=1, # LEARN RM EVERY EPISODE
                # ep_iterations=args.train_num_steps,
            )
            data.extend(d)
            if env.observable_RM==2: # use RM_learned
                pos_runs, neg_runs = [], []
                d_run = [env.lbl_of(s['s']) for s in d[0]]
                d_run.append(env.lbl) # fix last element that got truncated
                (pos_runs if d[0][-1]['done'] else neg_runs).append(d_run)
                all_pos_runs.extend(pos_runs); all_neg_runs.extend(neg_runs)
                if update_sample(env.RM_learned, sample, pos_runs=pos_runs, neg_runs=neg_runs): # check if new counterexamples
                    env.RM_learned = learn_RM(env.RM_learned.alphabet, sample, pos_runs=all_pos_runs, neg_runs=all_neg_runs, start_N=len(env.RM_learned.states)) # relearn RM
                    env.RM_learned.export_as_visualization_dot(
                        output_file="learned.dot",
                        keep_states=True, keep_alphabet=True,
                        group_separator=r';',
                    )
                    Q = args.parse_Q(args, env, args.alpha) # reset Q
        if args.path_train_save is not None:
            with open(args.path_train_save, 'wb') as f: pickle.dump(data, f)
    if hasattr(args, 'path_model_save') and args.path_model_save is not None:
        np.save(args.path_model_save, Q.w)
    
    logging.info("EVALUATION")
    if hasattr(args, 'seed'):
        env.seed(args.seed)
        set_seed(args.seed)
    data = run(
        evaluate, (rend_env if args.eval_render else env), Q,
        # tot_iterations=args.eval_tot_steps,
        episodes=args.eval_num_eps,
        ep_iterations=args.eval_num_steps,
    )
    if args.path_eval_save is not None:
        with open(args.path_eval_save, 'wb') as f: pickle.dump(data, f)


def create_parser_SAC(parser=None):
    parser = create_parser(parser,
        name='SAC',
        description="""
            Soft Actor Critic.
            Note that --discrete-actions=0 is required.
        """,
        epilog="""
            SAC parameters are fixed to
            learning_rate = 0.0025,
            buffer_size = 10000,
            learning_starts = 1000.
        """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(main=run_SAC)

    parser.add_argument('--save-model', metavar="PATH",
        dest='path_model_save',
        help="Save the trained model (.zip)",
    )
    parser.add_argument('--load-model', metavar="PATH",
        dest='path_model_load',
        help="Load a trained model",
    )

    return parser
def run_SAC(args:argparse.Namespace):
    env = args.parse_env(args)

    from sac import SoftActorCritic as SAC
    from utils import RenderWrapper

    import logging
    logging.basicConfig(level=logging.DEBUG)
    rend_env = RenderWrapper(env, fps=120)
    model = SAC(
        env=env,
        path=args.path_train_save,
        total_timesteps=args.train_tot_steps,
        seed=args.seed,
        render=rend_env if args.train_render else False,
    )
    if args.path_model_load is not None:
        logging.info("LOADING TRAINED MODEL")
        model.load(path=args.path_model_load)
    else:
        logging.info("TRAINING")
        model.train()
    if args.path_model_save is not None:
        model.load(path=args.path_model_save)
    logging.info("EVALUATION")
    model.evaluate(
        path=args.path_eval_save,
        num_eps=args.eval_num_eps,
        num_steps=args.eval_num_steps,
        render=rend_env if args.eval_render else False,
    )


# -- main -------------------------------------------------------------------- #

def create_parser_main(parser=None) -> argparse.ArgumentParser:
    parser = create_parser(
        # description=r"""
        #     Generates a chart.
        # """,
        # epilog=r"""
        #     To use this in latex:
        #     \immediate\write18{\unexpanded{python3 """+sys.argv[0]+""" -o figure.tex --width '0.5\linewidth'}}
        #     \input{figure.tex}
        # """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(main=print_help)
    create_parser_env(parser)

    group_train = parser.add_argument_group(title="training parameters")
    group_train.add_argument('--seed', metavar="INT",
        type=int, default=None,
        help="seed for generating random numbers",
    )
    group_train.add_argument('--train-iterations', metavar="INT",
        dest='train_tot_steps', type=int, default=int(1e5),
        help="total timesteps to train on (cumulative between episodes) (default 100000)",
    )
    group_train.add_argument('--eval-episodes', metavar="INT",
        dest='eval_num_eps', type=int, default=int(100),
        help="total evaluation episodes (default 100)",
    )
    group_train.add_argument('--eval-iterations', metavar="INT",
        dest='eval_num_steps', type=int, default=int(1e3),
        help="max timesteps per evaluation episode (default 1000)",
    )
    group_train.add_argument('--train-render',
        dest='train_render', action='store_true',
        help="render the environment at 120 fps during training",
    )
    group_train.add_argument('--eval-render',
        dest='eval_render', action='store_true',
        help="render the environment at 120 fps during evaluation",
    )
    group_train.add_argument('--save-train', metavar="PATH",
        dest='path_train_save',
        help="Save the training logs as a pickle file (.pkl)",
    )
    group_train.add_argument('--save-eval', metavar="PATH",
        dest='path_eval_save',
        help="Save the evaluation logs as a pickle file (.pkl)",
    )
    
    subparsers_algo = parser.add_subparsers(metavar="ALGO",
        title="algorithms",
        # required=True, # not working
    	help="algorithm to use",
    )
    create_parser_NStepSarsa(subparsers_algo)
    create_parser_SarsaLambda(subparsers_algo)
    # create_parser_SAC(subparsers_algo)
    return parser



if __name__ == '__main__':
	parser = create_parser_main()
	args = parser.parse_args()
	# args = parser.parse_args(['NStepSarsa', '-h'])
	args.main(args)