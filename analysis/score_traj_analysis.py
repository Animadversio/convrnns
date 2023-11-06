
import numpy as np
import matplotlib.pyplot as plt
import torch
from os.path import join
expdir = r"F:\insilico_exps\convrnn_Evol\rgc_intermediate-conv7_dyn"

savedict = np.load(join(expdir, "scores_conv7_chan000_T08.npz"))
#%%
list(savedict)
#%%
plt.scatter(savedict["generations"], savedict["scores"])
plt.show()
#%%
savedict = np.load(join(expdir, "scores_conv7_chan002_T07.npz"))
for T in range(savedict["scores_dyn"].shape[1]):
    mean_score = [np.mean(savedict["scores_dyn"][savedict["generations"]==gen, T])
                    for gen in range(savedict["generations"].max()+1)]
    std_score = [np.std(savedict["scores_dyn"][savedict["generations"]==gen, T])
                    for gen in range(savedict["generations"].max()+1)]
    # compute mean score for each gen
    plt.plot(range(savedict["generations"].max()+1), mean_score, label=f"T={T}")
    # plt.scatter(savedict["generations"], savedict["scores_dyn"][:, T], label=f"T={T}")
    # plt.scatter(savedict["generations"], savedict["scores_dyn"])
plt.legend()
plt.show()

#%%
