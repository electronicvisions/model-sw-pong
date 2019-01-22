import matplotlib as mpl
import gzip
import matplotlib.pyplot as plt
import cPickle as cp
import numpy as np
import argparse
import os
import tqdm
mpl.use("Agg")
plt.style.use(['seaborn-whitegrid'])
mpl.rc("font", family="Times New Roman", size=30)

parser = argparse.ArgumentParser(
    description="Plot mean reward vs. iterations for a number of experiments.")
parser.add_argument("--folder",
                    type=str,
                    required=True,
                    help="Folder containing experiment folders.")
parser.add_argument("--plot-max",
                    type=float,
                    default=1.0,
                    help="Maximum reward to plot.")
args = parser.parse_args()
plt.figure(figsize=(15, 10))
all_rewards = None
i = 0
for _, dirnames, _ in os.walk(args.folder):
    for direc in tqdm.tqdm(dirnames):
        for fname in os.listdir(os.path.join(args.folder, direc)):
            if not fname == "data.pkl.gz":
                continue
            i += 1
            print(fname)
            with gzip.open(os.path.join(args.folder, direc, fname), "r") as f:
                mean_rewards, _, _, runs = cp.load(f)
                masked = np.ma.masked_less(mean_rewards, 0)
                print masked
                mean_mean_rewards = masked.mean(axis=1)
                plt.plot(runs, mean_mean_rewards, "b", alpha=0.2)
                if all_rewards is None:
                    all_rewards = mean_mean_rewards
                else:
                    all_rewards += mean_mean_rewards
if i is 0:
    raise RuntimeError("No experiments found!")
all_rewards /= i
print runs, all_rewards
plt.plot(runs, all_rewards, "b")
plt.grid(color="black", alpha=1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.title("Simulation")
plt.ylim(0, args.plot_max)
plt.xlabel("Run #")
plt.ylabel("Mean Reward")
plt.savefig(os.path.join(args.folder, "mean_rewards.pdf"))
