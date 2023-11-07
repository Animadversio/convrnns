"""Analyzing results from evolution experiments"""
import re
import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from circuit_toolkit.plot_utils import make_grid, make_grid_T, make_grid_np, show_image_without_frame, saveallforms
from tqdm import trange
rootdir = r"F:\insilico_exps\convrnn_Evol\fasrc_runs"
layerdir = r"F:\insilico_exps\convrnn_Evol\fasrc_runs\rgc_intermediate-conv8_timewdw"

def extract_score_stats(score_data):
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
    # return a dict of stats
    return dict(mean_score_per_gen=mean_score_per_gen,
                std_score_per_gen=std_score_per_gen,
                mean_score_per_gen_dyn=mean_score_per_gen_dyn,
                final_score=final_score,
                max_score=max_score,
                max_gen=max_gen,
                max_score_dyn=max_score_dyn,
                mean_score_std_per_gen=mean_score_std_per_gen,
                final_score_dyn=final_score_dyn)


#%%
model_name = "rgc_intermediate"
# for layernum in range(4, 10+1):
layernum = 8
layername = f"conv{layernum}"
layerdir = rf"F:\insilico_exps\convrnn_Evol\fasrc_runs\rgc_intermediate-{layername}_timewdw"
T_active = layernum  # start receiving input at this timepoint
for channum in range(40):
    # for channum in range(50):
    score_traj_mat = defaultdict(list)
    score_traj_dyn_mat = defaultdict(list)
    max_score_mat = defaultdict(list)
    max_score_dyn_mat = defaultdict(list)
    final_score_mat = defaultdict(list)
    final_score_dyn_mat = defaultdict(list)
    # for T in range(T_active, 16 + 1):
    # parse these type of patterns
    BG_fns = list(Path(layerdir).glob(rf"scores_{layername}_chan{channum:03d}_T{T_active:02d}-T16_CholCMA-BigGAN_*.npz"))
    FC_fns = list(Path(layerdir).glob(rf"scores_{layername}_chan{channum:03d}_T{T_active:02d}-T16_CholCMA-fc6_*.npz"))
    try:
        assert len(list(BG_fns)) == 5
        assert len(list(FC_fns)) == 5
    except AssertionError:
        print(f"Missing files for {layername} chan{channum} T{T_active}-T16 in either BG or FC")

    final_score_dyn_col = defaultdict(list)
    for repi in range(5):
        BGdata = np.load(BG_fns[repi], allow_pickle=True)
        FCdata = np.load(FC_fns[repi], allow_pickle=True)
        score_dict_BG = extract_score_stats(BGdata)
        score_dict_FC = extract_score_stats(FCdata)
        final_score_dyn_col["BigGAN"].append(score_dict_BG["final_score_dyn"])
        final_score_dyn_col["DeePSim"].append(score_dict_FC["final_score_dyn"])
    for label in final_score_dyn_col:
        final_score_dyn_col[label] = np.array(final_score_dyn_col[label])

    plt.plot(range(T_active, 16+1), final_score_dyn_col["DeePSim"].T, label="DeePSim", color="blue")
    plt.plot(range(T_active, 16+1), final_score_dyn_col["BigGAN"].T, label="BigGAN", color="red")
    plt.legend()
    plt.title(f"Layer {layernum} Channel {channum}")
    plt.xlabel("Time steps (10ms)")
    plt.ylabel("Activation")
    # plt.savefig(join(layerdir, f"Layer{layernum}_chan{channum}_finalscoredyn.png"))
    plt.show()

#%%

#%%
plt.plot(range(T_active, 16+1), score_dict_FC["final_score_dyn"], label="FC")
plt.plot(range(T_active, 16+1), score_dict_BG["final_score_dyn"], label="BG")
plt.legend()
plt.title(f"Layer {layernum} Channel {channum}")
plt.xlabel("Time steps (10ms)")
plt.ylabel("Activation")
# plt.savefig(join(layerdir, f"Layer{layernum}_chan{channum}_finalscoredyn.png"))
plt.show()
