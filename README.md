
# CS394R: Reinforcement Learning: Theory and Practice -- Spring 2022 -- Project

https://www.cs.utexas.edu/~pstone/Courses/394Rspring22/assignments/project.html

# Installing

We used python 3.7 and the libraries in `requirements.txt`.

```sh
pip install --upgrade -r requirements.txt
```

# Running

Help messages are available:
```sh
./main.py -h
./main.py NStepSarsa -h
./main.py NStepSarsa TileCoding -h
./main.py NStepSarsa Network -h
./main.py SarsaLambda -h
./main.py SAC -h
# etc
```

Example of runs:

```sh
./main.py --obs=1 --RM=0 --gamma=1 NStepSarsa --n=8 --alpha=0.05 TileCoding
./main.py --obs=1 --RM=0 --gamma=1 NStepSarsa --n=8 --alpha=0.0001 Network --RMenc=NNs
./main.py --obs=1 --RM=0 --gamma=1 SarsaLambda --lambda=0.95 --alpha=0.1 TileCoding
```