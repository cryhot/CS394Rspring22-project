#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import argparse

argparse._SubParsersAction


# # create the top-level parser
# parser = argparse.ArgumentParser(prog='PROG')
# parser.add_argument('--foo', action='store_true', help='foo help')
# # parser.set_defaults(p='-')
# subparsers = parser.add_subparsers(help='sub-command help')

# # create the parser for the "a" command
# parser_a = subparsers.add_parser('--a', help='a help')
# parser_a.add_argument('bar', type=int, help='bar help')
# # parser_a.set_defaults(p='a')

# # create the parser for the "b" command
# parser_b = subparsers.add_parser('--b', help='b help')
# parser_b.add_argument('--baz', choices='XYZ', help='baz help')
# parser_b.set_defaults(p='b')

# # parse some argument lists
# print(subparsers.__class__)
# # parser.parse_args(['a', '12'])
# args = parser.parse_args()

# print(args)
# exit()


# == helper functions ======================================================== #
def create_parser(parser=None, name=None, **kwargs) -> argparse.ArgumentParser:
    """Return either the existing parser, either a fresh parser or subparser."""
    if parser is None:
        parser = argparse.ArgumentParser(**kwargs)
    elif isinstance(parser, argparse._SubParsersAction):
        if 'description' in kwargs: kwargs['help'] = kwargs.pop('description')
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
            description=r"""
                Create a MountainCar environment.
            """,
        )
    parser.set_defaults(parser=parser)
    parser.set_defaults(parse_env=parse_env)

    group_env = parser.add_argument_group(title="environment parameters")

    group_env.add_argument('--RM', metavar="ID",
        type=int, default='0',
        help = "Preconfigured Reward machine and Stops: 0 is (R=-1, Ra=0, Rb=0), 1 is (R=-0.1, Ra=10, Rb=100)",
        choices=range(2),
    )

    group_env.add_argument('--observable-RM',
        dest='observable_RM', #required=True,
        type=int, choices=[0,1],
        help = "Makes the Reward Machine observable by the agent",
    )

    group_env.add_argument('--gamma', metavar="FLOAT",
        type=float, default=1,
        help = "discount factor",
    )

    return parser
def parse_env(args:argparse.Namespace):
    from mountain_car import MountainCarEnvWithStops as MountainCar
    from parameters import RM
    if 'RM' in args:
        RM_args = RM[args.RM]
    env = MountainCar(
        gamma=args.gamma,
        observable_RM=args.observable_RM,
        **RM_args,
    )
    return env

# -- ValueFunction ----------------------------------------------------------- #

def create_parser_TileCoding(parser=None):
    parser = create_parser(parser,
        name='TileCoding',
        description=r"""
            Q function approximator with Tile Coding.
        """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(parse_Q=parse_TileCoding)
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
                round(env.observation_space.high[-1])+1,
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
        description=r"""
            Q function approximator with Tile Coding.
        """,
    )
    parser.set_defaults(parser=parser)
    parser.set_defaults(parse_Q=parse_Network)

    group_NN = parser.add_argument_group(title="Neural Network parameters")

    group_NN.add_argument('--RMenc', metavar="ENCODING",
        choices=['OneHot', 'Linear', 'NNs'],
        help = "NNs: one NN per RM state; otherwise, encode RM state as inpute neuron (1 or one-hot).",
    )
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
                            round(env.observation_space.high[-1])+1,
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
        description=r"""
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
        # help="Q-function approximator to use",
    )
    create_parser_TileCoding(subparsers_Q)
    create_parser_Network(subparsers_Q)

    return parser
def run_NStepSarsa(args:argparse.Namespace):
    if not hasattr(args, 'parse_Q'):
        args.parser.error('the value function approximator Q is required')
    env = args.parse_env(args)
    Q = args.parse_Q(args, env, args.alpha)

    from rl.algo import n_step_Sarsa, evaluate
    from rl.algo import run
    from utils import RenderWrapper

    import logging
    logging.basicConfig(level=logging.DEBUG)
    rend_env = RenderWrapper(env, fps=120)
    run(
        n_step_Sarsa, rend_env, Q, n=args.n, alpha=args.alpha,
        # tot_iterations=train_total_timesteps,
        # episodes=eval_num_eps,
        # ep_iterations=eval_num_steps,
    )

def create_parser_SarsaLambda(parser=None):
    parser = create_parser(parser,
        name='SarsaLambda',
        description=r"""
            True Online Sarsa(lambda).
        """,
    )
    create_parser_env(parser)
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
    Q = args.parse_Q(args, env, args.alpha)

    from rl.algo import SarsaLambda, evaluate
    from rl.algo import run
    from utils import RenderWrapper

    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    rend_env = RenderWrapper(env, fps=120)
    run(
        SarsaLambda, rend_env, Q, lam=args.lam, alpha=args.alpha,
        # tot_iterations=train_total_timesteps,
        # episodes=eval_num_eps,
        # ep_iterations=eval_num_steps,
    )

    return parser


def create_parser_SAC(parser=None):
    parser = create_parser(parser,
        name='SAC',
        description=r"""
            Episodic n-step Semi-gradient Sarsa.
        """,
    )
    create_parser_env(parser)
    parser.set_defaults(parser=parser)
    parser.set_defaults(main=print_help)

    return parser


# -- main -------------------------------------------------------------------- #

def create_parser_main(parser=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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
    
    subparsers_algo = parser.add_subparsers(metavar="ALGO",
        title="algorithms",
        # required=True, # not working
    	help="algorithm to use",
    )
    create_parser_NStepSarsa(subparsers_algo)
    create_parser_SarsaLambda(subparsers_algo)
    create_parser_SAC(subparsers_algo)
    return parser



if __name__ == '__main__':
	parser = create_parser_main()
	args = parser.parse_args()
	# args = parser.parse_args(['NStepSarsa', '-h'])
	args.main(args)