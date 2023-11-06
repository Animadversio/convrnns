"""Analyzing results from evolution experiments"""
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
rootdir = r"F:\insilico_exps\convrnn_Evol\fasrc_runs"
model_name = "rgc_intermediate"

layernum = 8
layername = f"conv{layernum}"
layerdir_BG = join(rootdir, rf"{model_name}-{layername}_dyn_BigGAN")
layerdir_FC = join(rootdir, rf"{model_name}-{layername}_dyn")

T_active = layernum  # start receiving input at this timepoint
channum = 40
score_traj_mat = defaultdict(list)
score_traj_dyn_mat = defaultdict(list)
max_score_mat = defaultdict(list)
max_score_dyn_mat = defaultdict(list)
final_score_mat = defaultdict(list)
final_score_dyn_mat = defaultdict(list)
for T in range(T_active, 16 + 1):
    # parse these type of patterns
    BG_fns = list(Path(layerdir_BG).glob(rf"scores_{layername}_chan{channum:03d}_T{T:02d}_CholCMA-BigGAN_*.npz"))
    FC_fns = list(Path(layerdir_FC).glob(rf"scores_{layername}_chan{channum:03d}_T{T:02d}.npz"))
    assert len(list(BG_fns)) == 1
    assert len(list(FC_fns)) == 1
    BGdata = np.load(BG_fns[0], allow_pickle=True)
    FCdata = np.load(FC_fns[0], allow_pickle=True)
    for label, score_data in zip(("BigGAN", "DeePSim"),
                                 (BGdata, FCdata)):
        scores = score_data['scores']
        gennums = score_data['generations']
        scores_dyn = score_data['scores_dyn']
        mean_score_per_gen = np.array([np.mean(scores[gennums == gen]) for gen in range(gennums.max()+1)])
        std_score_per_gen = np.array([np.std(scores[gennums == gen]) for gen in range(gennums.max()+1)])
        mean_score_per_gen_dyn = np.array([np.mean(scores_dyn[gennums == gen], axis=0) for gen in range(gennums.max()+1)])
        final_score = mean_score_per_gen[-1]
        max_score = mean_score_per_gen.max()
        max_gen = mean_score_per_gen.argmax()
        mean_score_std_per_gen = std_score_per_gen.mean()
        final_score_dyn = mean_score_per_gen_dyn[-1]
        max_score_dyn = mean_score_per_gen_dyn.max(axis=0)
        # print(max_score)
        # print(max_score_dyn)
        score_traj_mat[label].append(mean_score_per_gen)
        score_traj_dyn_mat[label].append(mean_score_per_gen_dyn)
        final_score_mat[label].append(final_score)
        final_score_dyn_mat[label].append(final_score_dyn)
        max_score_mat[label].append(max_score)
        max_score_dyn_mat[label].append(max_score_dyn)
for label in max_score_mat:
    score_traj_mat[label] = np.array(score_traj_mat[label])
    score_traj_dyn_mat[label] = np.array(score_traj_dyn_mat[label])
    final_score_mat[label] = np.array(final_score_mat[label])
    final_score_dyn_mat[label] = np.array(final_score_dyn_mat[label])
    max_score_mat[label] = np.array(max_score_mat[label])
    max_score_dyn_mat[label] = np.array(max_score_dyn_mat[label])
#%%
plt.subplots(1, 2, figsize=(11, 5))
plt.subplot(1, 2, 1)
sns.heatmap(final_score_dyn_mat["BigGAN"],
            xticklabels=range(T_active, 17),
            yticklabels=range(T_active, 17),
            cmap="inferno", annot=True, fmt=".0f")
plt.ylabel("Optimized Timepoint")
plt.xlabel("Response Time")
plt.title("BigGAN")
plt.axis("image")
plt.subplot(1, 2, 2)
sns.heatmap(final_score_dyn_mat["DeePSim"],
            xticklabels=range(T_active, 17),
            yticklabels=range(T_active, 17),
            cmap="inferno", annot=True, fmt=".0f")
plt.ylabel("Optimized Timepoint")
plt.xlabel("Response Time")
plt.title("DeePSim")
plt.axis("image")
plt.suptitle(f"Final Score per Timepoint\n{layername} chan{channum:03d} T={T_active}-16")
plt.tight_layout()
plt.show()

#%%
plt.subplots(2, 1, figsize=(11, 5))
plt.subplot(2, 1, 1)
sns.heatmap(score_traj_mat["BigGAN"], yticklabels=range(T_active, 17),
            cmap="inferno", )  # annot=True, fmt=".0f")
plt.title("BigGAN")
plt.ylabel("Optimized Timepoint")
plt.xlabel("Blocks")
plt.subplot(2, 1, 2)
sns.heatmap(score_traj_mat["DeePSim"], yticklabels=range(T_active, 17),
            cmap="inferno", )  # annot=True, fmt=".0f")
plt.title("DeePSim")
plt.ylabel("Optimized Timepoint")
plt.xlabel("Blocks")
plt.suptitle(f"Score Trajectory per Timepoint\n{layername} chan{channum:03d} T={T_active}-16")
plt.tight_layout()
plt.show()
#%%
plt.subplots(2, 1, figsize=(11, 5))
plt.subplot(2, 1, 1)
sns.heatmap(score_traj_mat["BigGAN"] / score_traj_mat["BigGAN"].max(axis=1, keepdims=True),
            yticklabels=range(T_active, 17),
            cmap="inferno", )  # annot=True, fmt=".0f")
plt.title("BigGAN")
plt.ylabel("Optimized Timepoint")
plt.xlabel("Blocks")
plt.subplot(2, 1, 2)
sns.heatmap(score_traj_mat["DeePSim"] / score_traj_mat["DeePSim"].max(axis=1, keepdims=True),
            yticklabels=range(T_active, 17),
            cmap="inferno", )  # annot=True, fmt=".0f")
plt.title("DeePSim")
plt.ylabel("Optimized Timepoint")
plt.xlabel("Blocks")
plt.suptitle(f"Normalized Score Trajectory per Timepoint\n{layername} chan{channum:03d} T={T_active}-16")
plt.tight_layout()
plt.show()
#%%
def extract_score_stats(score_data):
    scores = score_data['scores']
    gennums = score_data['generations']
    scores_dyn = score_data['scores_dyn']
    mean_score_per_gen = np.array([np.mean(scores[gennums == gen]) for gen in range(gennums.max()+1)])
    std_score_per_gen = np.array([np.std(scores[gennums == gen]) for gen in range(gennums.max()+1)])
    mean_score_per_gen_dyn = np.array([np.mean(scores_dyn[gennums == gen]) for gen in range(gennums.max()+1)])
    final_score = mean_score_per_gen[-1]
    max_score = mean_score_per_gen.max()
    max_gen = mean_score_per_gen.argmax()
    mean_score_std_per_gen = std_score_per_gen.mean()
    final_score_dyn = mean_score_per_gen_dyn[-1]
    max_score_dyn = mean_score_per_gen_dyn.max()
    # return a dict of stats
    return dict(mean_score_per_gen=mean_score_per_gen,
                std_score_per_gen=std_score_per_gen,
                mean_score_per_gen_dyn=mean_score_per_gen_dyn,
                final_score=final_score,
                max_score=max_score,
                max_gen=max_gen,
                mean_score_std_per_gen=mean_score_std_per_gen,
                final_score_dyn=final_score_dyn)


#%%
