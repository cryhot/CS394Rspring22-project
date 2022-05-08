import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]

if __name__ == '__main__':
    window = 50

    train_filenames = os.listdir("logs/train/")
    train_filenames.sort()
    eval_filenames = os.listdir("logs/eval/")
    eval_filenames.sort()

    train_data = []
    for file in train_filenames[:3]:
        print(file)
        with open("logs/train/"+file, 'rb') as handle:
            train_data.append(pickle.load(handle))

    eval_data = []
    for file in eval_filenames[:3]:
        with open("logs/eval/"+file, 'rb') as handle:
            eval_data.append(pickle.load(handle))

    train = []
    for t, data in enumerate(train_data):
        # Plot: Reward & number of steps per episode progression
        eps = []
        tr = []
        for ep in np.unique(data["episode"]):
            eps.append(data[data["episode"]==ep].shape[0])
            tr.append(data[data["episode"]==ep]["r"].sum())
        train.append([moving_average(tr,5),eps])

    for t, data in enumerate(eval_data):
        # Plot: Reward & number of steps per episode progression
        eps = []
        r = []
        for ep in np.unique(data["episode"]):
            eps.append(data[data["episode"]==ep].shape[0])
            r.append(data[data["episode"]==ep]["r"].mean())

        print(eval_filenames[t])
        print("mean eval r: ", np.array(r).mean())
        print("mean eval eps: ", np.array(eps).mean())


    plt.figure()
    plt.plot(train[0][0],label="seed=")
    plt.plot(train[1][0],label="seed=")
    plt.plot(train[2][0],label="seed=")
    plt.xlabel("Training iteration")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("training_rew.png")

    plt.figure()
    plt.plot(train[0][1],label="seed=1")
    plt.plot(train[1][1],label="seed=2")
    plt.plot(train[2][1],label="seed=3")
    plt.xlabel("Training iteration")
    plt.ylabel("Number of steps per episode")
    plt.legend()
    plt.savefig("training_eps.png")