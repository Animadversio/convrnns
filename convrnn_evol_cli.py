import os
from os.path import join
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from convrnns.utils.loader import get_restore_vars, MODEL_TO_KWARGS
from convrnns.models.model_func import model_func
import sys
sys.path.append("/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Github/circuit_toolkit")
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor, ToPILImage
from easydict import EasyDict as edict
# from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.Optimizers import CholeskyCMAES
from circuit_toolkit.GAN_utils import upconvGAN
# from circuit_toolkit.layer_hook_utils import get_module_names

def normalize_ims(x):
    # ensures that images are between 0 and 1
    x = x.astype(np.float32)
    assert np.amin(x) >= 0
    if np.amax(x) > 1:
        assert np.amax(x) <= 255
        print("Normalizing images to be between 0 and 1")
        x /= 255.0

    # this is important to preserve on new stimuli, since the models were trained with these image normalizations
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - imagenet_mean) / imagenet_std
    return x


def print_keys_recursive(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + f'Key: {key}')
        if isinstance(value, dict):
            print_keys_recursive(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print('  ' * (indent + 1) + f'Shape: {value.shape}, Dtype: {value.dtype}')
        else:
            print('  ' * (indent + 1) + f'Value: {value}')

# import sys
# sys.path.append("/home/biw905/Github/convrnns")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="rgc_intermediate",
    help="Which model to load.",
    required=True,
)
parser.add_argument(
    "--layername",
    type=str,
    default="imnetds",
    help="Which layers of the model to output. Pass in a comma separated list if you want multiple layers.",
)
parser.add_argument(
    "--chan_start",
    type=int,
    default=0,
    help="channel start",
)
parser.add_argument(
    "--chan_end",
    type=int,
    default=20,
    help="channel end",
)
cli_args = parser.parse_args()
channel_range = (cli_args.chan_start, cli_args.chan_end)
args = edict({
    # "model_name": "rgc_intermediate", #'rgc_intermediate',  # Required argument, so no default value is effectively used
    # "out_layers": 'conv2',
    "model_name": cli_args.model_name,
    "gpu": None,
    "ckpt_dir": "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Github/convrnns/ckpts", # "/home/biw905/Github/convrnns/ckpts",#
    "image_pres": "default",
    "times": None,
    "image_off": None,
    "include_all_times": False,
    "include_logits": False
})

G = upconvGAN("fc6").eval().cuda()
G.requires_grad_(False)

num_batch = 20
inputs = tf.placeholder(tf.float32, shape=[num_batch, 224, 224, 3])
layername = cli_args.layername
out_layers = [layername]
# out_layers = args.out_layers.split(",")
model_kwargs = MODEL_TO_KWARGS[args.model_name]
y = model_func(
    inputs=inputs,
    out_layers=out_layers,
    image_pres=args.image_pres,
    times=args.times,
    image_off=args.image_off,
    include_all_times=args.include_all_times,
    include_logits=args.include_logits,
    **model_kwargs
)
sess = tf.Session()
CKPT_PATH = join(args.ckpt_dir, "{}/model.ckpt".format(args.model_name))
restore_vars = get_restore_vars(CKPT_PATH)
tf.train.Saver(var_list=restore_vars).restore(sess, CKPT_PATH)

#%% time bin score
steps = 100
savedir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/convrnn_Evol"
expdir = join(savedir, f"{args.model_name}-{layername}_dyn")
os.makedirs(expdir, exist_ok=True)
# G = upconvGAN("fc6").cuda().eval()
for chan_id in range(channel_range[0], channel_range[1]):
    for time_idx in range(len(y[layername])):  # todo this is the shape of the output
        print("%s Channel %d, time bin id %d" % (layername, chan_id, time_idx))
        # main loop of Evolution.
        new_codes = np.random.randn(1, 4096)
        optimizer = CholeskyCMAES(space_dimen=4096, init_code=new_codes, init_sigma=3.0,)
        savedict = dict({"scores": [], "scores_dyn": [], "codes": [], "generations": []})
        best_imgs = []
        for block_i in range(steps):
            n_imgs = len(new_codes)
            latent_code = torch.from_numpy(np.array(new_codes)).float()
            savedict["codes"].append(new_codes)
            imgs = G.visualize(latent_code.cuda()).detach().cpu()
            # imgs = G.visualize(latent_code).detach()
            imgs_pp = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
            imgs_pp_np = imgs_pp.numpy().transpose(0, 2, 3, 1)
            imgs_pp_np = normalize_ims(imgs_pp_np)
            # zero pad to batch size
            scores_dyn = []
            scores = []
            for n_csr in range(0, n_imgs, num_batch):
                if n_imgs < num_batch:
                    imgs_pp_batch = np.concatenate([imgs_pp_np, np.zeros((num_batch - n_imgs, 224, 224, 3))], axis=0)
                else:
                    imgs_pp_batch = imgs_pp_np[n_csr:n_csr+num_batch, :]
                # Record activations
                # scores = scorer.score_tsr(imgs)
                y_eval = sess.run(y, feed_dict={inputs: imgs_pp_batch})
                # Calculate scores
                time_pnts = list(y_eval[layername].keys()) # [Tpnts,]
                acttsr = np.array([*y_eval[layername].values()]) # [Tpnts, batch, H, W, C, ]
                Hcent, Wcent = acttsr.shape[2] // 2, acttsr.shape[3] // 2
                scores_dyn_batch = acttsr[:, :min(n_imgs, num_batch), Hcent, Wcent, chan_id]  # [T, n_imgs,]
                scores_batch = scores_dyn_batch[time_idx, :]  # [n_imgs,]
                scores_dyn.append(scores_dyn_batch)
                scores.append(scores_batch)

            scores = np.concatenate(scores, axis=0)
            scores_dyn = np.concatenate(scores_dyn, axis=1)
            # Perform CMA updates
            new_codes = optimizer.step_simple(scores, new_codes, )
            print("step %d score %.3f (%.3f) (norm %.2f )" % (
                    block_i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
            savedict["scores"].append(scores)
            savedict["scores_dyn"].append(scores_dyn.transpose(1, 0)) # batch, time
            savedict["generations"].append([block_i]*n_imgs)
            best_imgs.append(imgs[np.argmax(scores), :, :, :])
            torch.cuda.empty_cache()

        for k in savedict:
            savedict[k] = np.concatenate(savedict[k], axis=0)
        savedict["time_pnts"] = time_pnts
        savedict["Hcent"] = Hcent
        savedict["Wcent"] = Wcent
        time_wdw_str = "%02d" % time_pnts[time_idx]
        plt.imsave(join(expdir, f"{layername}_chan{chan_id:03d}_T{time_wdw_str}_lastgen.png"),
                   imgs.permute(0, 2, 3, 1).numpy()[0], )
        np.savez(join(expdir, f"scores_{layername}_chan{chan_id:03d}_T{time_wdw_str}.npz"),
                 **savedict)
        mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
        mtg_exp.save(join(expdir, f"besteachgen_{layername}_chan{chan_id:03d}_T{time_wdw_str}.jpg" ))
        mtg = ToPILImage()(make_grid(imgs, nrow=7))
        mtg.save(join(expdir, f"lastgen_{layername}_chan{chan_id:03d}_T{time_wdw_str}.jpg"))

#%%
sess.close()

