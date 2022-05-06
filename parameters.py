
import numpy as np

RM = [
    dict(
        trans_reward = -1,
        stops=[ # (state_center, state_width, reward)
            ([ 0.525, 0.035], [0.15, np.infty], 0),
            ([-0.5,   0.   ], [0.2,  0.02    ], 0),
        ],
    ),
    dict(
        trans_reward = -0.1,
        stops=[ # (state_center, state_width, reward)
            ([ 0.525, 0.035], [0.15, np.infty],  10),
            ([-0.5,   0.   ], [0.2,  0.02    ], 100),
        ],
    ),
]