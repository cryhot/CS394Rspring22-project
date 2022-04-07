#!/usr/bin/env python3
from mountain_car import MountainCarEnvWithStops as MountainCar

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = MountainCar()
    s, done = env.reset(), False
    env.render()

    while not done:
        a = 1
        s, r, done, _ = env.step(a)
        env.render()
