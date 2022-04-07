from mountain_car import Continuous_MountainCarEnv_WithStops as mcar

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = mcar()
    s, done = env.reset(), False
    env.render()

    while not done:
        a = [0.]
        s, r, done, _ = env.step(a)
        env.render()
