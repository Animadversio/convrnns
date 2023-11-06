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
channum = 24
final_img_col = defaultdict(list)
final_score_col = defaultdict(list)
final_score_dyn_col = defaultdict(list)
for T in range(T_active, 16 + 1):
    # parse these type of patterns
    BG_fns = list(Path(layerdir_BG).glob(rf"scores_{layername}_chan{channum:03d}_T{T:02d}_CholCMA-BigGAN_*.npz"))
    FC_fns = list(Path(layerdir_FC).glob(rf"scores_{layername}_chan{channum:03d}_T{T:02d}.npz"))
    BGproto_fns = list(Path(layerdir_BG).glob(rf"conv{layernum}_chan{channum:03d}_T{T:02d}_lastgen_CholCMA-BigGAN_*.png"))
    FCproto_fns = list(Path(layerdir_FC).glob(rf"conv{layernum}_chan{channum:03d}_T{T:02d}_lastgen.png"))
    assert len(list(BG_fns)) == 1
    assert len(list(FC_fns)) == 1
    assert len(list(BGproto_fns)) == 1
    assert len(list(FCproto_fns)) == 1
    BGdata = np.load(BG_fns[0], allow_pickle=True)
    FCdata = np.load(FC_fns[0], allow_pickle=True)
    BGproto = plt.imread(BGproto_fns[0])
    FCproto = plt.imread(FCproto_fns[0])
    for label, score_data, protoimg in zip(("BigGAN", "DeePSim"),
                                 (BGdata, FCdata),
                                 (BGproto, FCproto)):
        scores = score_data['scores']
        gennums = score_data['generations']
        scores_dyn = score_data['scores_dyn']
        mean_score_per_gen = np.array([np.mean(scores[gennums == gen]) for gen in range(gennums.max()+1)])
        std_score_per_gen = np.array([np.std(scores[gennums == gen]) for gen in range(gennums.max()+1)])
        mean_score_per_gen_dyn = np.array([np.mean(scores_dyn[gennums == gen], axis=0) for gen in range(gennums.max()+1)])
        # final_score = mean_score_per_gen[-1]
        # max_score = mean_score_per_gen.max()
        # max_gen = mean_score_per_gen.argmax()
        # mean_score_std_per_gen = std_score_per_gen.mean()
        # final_score_dyn = mean_score_per_gen_dyn[-1]
        # max_score_dyn = mean_score_per_gen_dyn.max(axis=0)

        final_score_col[label].append(mean_score_per_gen[-1])
        final_img_col[label].append(protoimg[:,:,:3])
        final_score_dyn_col[label].append(mean_score_per_gen_dyn[-1])

for label in final_score_col:
    final_score_col[label] = np.array(final_score_col[label])
    final_score_dyn_col[label] = np.array(final_score_dyn_col[label])
#%%
from circuit_toolkit.plot_utils import make_grid, make_grid_T, make_grid_np, show_image_without_frame
BGmtg = make_grid_np(final_img_col["BigGAN"], nrow=len(final_img_col["BigGAN"]), padding=2)
FCmtg = make_grid_np(final_img_col["DeePSim"], nrow=len(final_img_col["BigGAN"]), padding=2)
Bothmtg = make_grid_np(final_img_col["BigGAN"] + final_img_col["DeePSim"],
                       nrow=len(final_img_col["BigGAN"]), padding=2)
#%%
show_image_without_frame(Bothmtg)
#%%
plt.imshow(Bothmtg)
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
sns.heatmap(np.vstack([final_score_col["BigGAN"],
                       final_score_col["DeePSim"]]),
            square=True, fmt=".0f", annot=True,
            yticklabels=["BigGAN", "DeePSim"],
            xticklabels=range(T_active, 17),
            cmap="inferno")
plt.show()
#%%
plt.subplots(1, 1, figsize=(6.5, 9.0))
sns.heatmap(np.vstack([final_score_dyn_col["BigGAN"],
                       final_score_dyn_col["DeePSim"]]),
            square=True, fmt=".0f", annot=True,
            yticklabels=[f"BigGAN T{T}" for T in range(T_active, 17)] +
                        [f"DeePSim T{T}" for T in range(T_active, 17)],
            xticklabels=range(T_active, 17),
            cmap="inferno")
plt.show()