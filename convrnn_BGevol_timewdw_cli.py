"""Using a slice of time as score instead of single time points"""
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
from circuit_toolkit.Optimizers import CholeskyCMAES, HessCMAES
from circuit_toolkit.Optimizers import fix_param_wrapper, concat_wrapper, label2optimizer
from circuit_toolkit.GAN_utils import upconvGAN, BigGAN_wrapper
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, one_hot_from_names, save_as_images)
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
import ast
# def parse_int_tuple(string, element_type=int, length=2):
#     try:
#         # This will safely evaluate the string into a Python literal
#         val = ast.literal_eval(string)
#         if (isinstance(val, tuple) and
#                 len(val) == length and
#         all(isinstance(num, element_type) for num in val)):
#             return val
#         else:
#             raise argparse.ArgumentTypeError("Each argument must be a tuple of two integers, e.g., (5, 16)")
#     except (ValueError, SyntaxError):
#         raise argparse.ArgumentTypeError("Each argument must be a tuple of two integers, e.g., (5, 16)")
def parse_int_tuple(string):
    try:
        x, y = map(int, string.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Input must be a comma-separated pair of integers.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="rgc_intermediate",
    help="Which model to load.",
)
parser.add_argument(
    "--layername",
    type=str,
    default="conv8",
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
parser.add_argument(
    "--steps",
    type=int,
    default=100,
    help="Evolution optimization steps in each run"
)
parser.add_argument(
    "--G",
    type=str,
    default="BigGAN",
    help="GAN used in Evolution process",
)
parser.add_argument("--optim",
    type=str,
    nargs='+',
    default=["CholCMA"], # "HessCMA",
    help="Evolution optimizer"
)
parser.add_argument("--timewdws",
    type=parse_int_tuple,
    nargs='+',
    default=[(1,16)],  # "HessCMA",
    help="time window counted in objective for running evolution, "
         "input as a list of tuples, no space between the int in the tuple. "
         "space allowed between tuples e.g. 1,16 5,16"
)
parser.add_argument(
    "--reps",
    type=int,
    default=1,
    help="Repetitions"
)
# cli_args = parser.parse_args(["--model_name", "rgc_intermediate", "--timewdws", "(1, 16)", "(5, 16)"])
cli_args = parser.parse_args()
cli_args = edict(vars(cli_args))
# cli_args = edict(steps=100, chan_start=0, chan_end=2, optim=["CholCMA"], G="BigGAN", layername="conv9", model_name="rgc_intermediate")
timewdws = cli_args.timewdws
steps = cli_args.steps
channel_range = (cli_args.chan_start, cli_args.chan_end)
method_col = cli_args.optim
reps = cli_args.reps
args = edict({
    # "model_name": "rgc_intermediate", #'rgc_intermediate',  # Required argument, so no default value is effectively used
    # "out_layers": 'conv2',
    "G": cli_args.G,
    "steps": cli_args.steps,
    "model_name": cli_args.model_name,
    "gpu": None,
    "ckpt_dir": "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Github/convrnns/ckpts", # "/home/biw905/Github/convrnns/ckpts",#
    "image_pres": "default",
    "times": None,
    "image_off": None,
    "include_all_times": False,
    "include_logits": False
})
# Loading GAN
if args.G == "BigGAN":
    BGAN = BigGAN.from_pretrained("biggan-deep-256")
    BGAN.eval().cuda()
    for param in BGAN.parameters():
        param.requires_grad_(False)
    G = BigGAN_wrapper(BGAN)
    G.visualize(torch.randn(30, 256).cuda())
    num_batch = 15  # default processing batch, smaller to avoid oom (BG takes a lot of GPU memory)
elif args.G == "fc6":
    G = upconvGAN("fc6").eval().cuda()
    G.requires_grad_(False)
    G.visualize(torch.randn(40, 4096).cuda())
    num_batch = 20  # default processing batch

# building computational graph in tensorflow.
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
#%%
min_active_time = min(y[layername].keys())  # key starts from 1
max_active_time = max(y[layername].keys())  # key ends with 16
# process the time windows
timewdws_processed = []
timewdws_idxs = []
for timewdw in timewdws:
    # limit the window to the actual activation time window.
    wdw_start = max(min_active_time, timewdw[0])
    wdw_end = min(max_active_time, timewdw[1])
    # this will be the printable tuple.
    timewdws_processed.append((wdw_start, wdw_end))
    # this will be the slice used for averaging the score.
    timewdws_idxs.append((wdw_start - min_active_time, wdw_end - min_active_time + 1))

print("Time windows under objectives", timewdws_processed)
print("Indices slice for each window", timewdws_idxs)
#%% time bin score
savedir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/convrnn_Evol"
expdir = join(savedir, f"{args.model_name}-{layername}_timewdw")
os.makedirs(expdir, exist_ok=True)
for chan_id in range(channel_range[0], channel_range[1]):
    for timewdw_proc, idxtuple in (
            zip(timewdws_processed, timewdws_idxs)):  # todo this is the shape of the output
        print("%s Channel %d, time bin T%d-T%d, averaging %d steps" % (layername, chan_id, timewdw_proc[0], timewdw_proc[1],
                                                                        idxtuple[1] - idxtuple[0]))
        time_wdw_str = "T%02d-T%02d" % (timewdw_proc[0], timewdw_proc[1])
        time_slice = slice(idxtuple[0], idxtuple[1])
        for rep in range(reps):
            if args.G == "BigGAN":
                fixnoise = 0.7 * truncated_noise_sample(1, 128)
                init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
            elif args.G == "fc6":
                init_code = np.random.randn(1, 4096)
            else:
                raise NotImplemented("Not recognized G type.")

            new_codes = init_code
            optimizer_col = [label2optimizer(methodlabel, init_code, args.G, Hdata=None) for methodlabel in method_col]
            RND = np.random.randint(1E5)
            for methodlab, optimizer in zip(method_col, optimizer_col):
                if args.G == "fc6":
                    methodlab += "-fc6"
                elif args.G == "BigGAN":
                    methodlab += "-BigGAN"
                # main loop of Evolution.
                savedict = dict({"scores": [], "scores_dyn": [], "generations": [], "best_codes": []}) # "mean_codes": [],
                best_imgs = []
                for block_i in range(steps):
                    n_imgs = len(new_codes)
                    latent_code = torch.from_numpy(np.array(new_codes)).float()
                    # savedict["mean_codes"].append(new_codes.mean(axis=0, keepdims=True))
                    imgs = G.visualize(latent_code.cuda()).detach().cpu()
                    # imgs = G.visualize(latent_code).detach()
                    imgs_pp = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
                    imgs_pp_np = imgs_pp.numpy().transpose(0, 2, 3, 1)
                    imgs_pp_np = normalize_ims(imgs_pp_np)
                    # zero pad to batch size
                    scores_dyn = []
                    scores = []
                    for n_csr in range(0, n_imgs, num_batch):
                        n_end = min(n_imgs, n_csr + num_batch)
                        if n_end - n_csr < num_batch:
                            imgs_pp_batch = np.concatenate([imgs_pp_np[n_csr:n_end, :], np.zeros((num_batch - (n_end - n_csr), 224, 224, 3))], axis=0)
                        else:
                            imgs_pp_batch = imgs_pp_np[n_csr:n_end, :]
                        # Record activations
                        # scores = scorer.score_tsr(imgs)
                        y_eval = sess.run(y, feed_dict={inputs: imgs_pp_batch})
                        # Calculate scores
                        time_pnts = list(y_eval[layername].keys()) # [Tpnts,]
                        acttsr = np.array([*y_eval[layername].values()]) # [Tpnts, batch, H, W, C, ]
                        Hcent, Wcent = acttsr.shape[2] // 2, acttsr.shape[3] // 2
                        scores_dyn_batch = acttsr[:, :(n_end - n_csr), Hcent, Wcent, chan_id]  # [T, n_imgs,]
                        scores_batch = scores_dyn_batch[time_slice, :].mean(axis=0)  # [n_imgs,]
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
                    savedict["best_codes"].append(new_codes[np.argmax(scores), :][np.newaxis])
                    torch.cuda.empty_cache()

                for k in savedict:
                    savedict[k] = np.concatenate(savedict[k], axis=0)
                savedict["time_pnts"] = time_pnts
                savedict["Hcent"] = Hcent
                savedict["Wcent"] = Wcent
                plt.imsave(join(expdir, f"{layername}_chan{chan_id:03d}_{time_wdw_str}_lastgen_{methodlab}_{RND}.png"),
                           imgs.permute(0, 2, 3, 1).numpy()[0], )
                np.savez(join(expdir, f"scores_{layername}_chan{chan_id:03d}_{time_wdw_str}_{methodlab}_{RND}.npz"),
                         **savedict)
                mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
                mtg_exp.save(join(expdir, f"besteachgen_{layername}_chan{chan_id:03d}_{time_wdw_str}_{methodlab}_{RND}.jpg" ))
                mtg = ToPILImage()(make_grid(imgs, nrow=7))
                mtg.save(join(expdir, f"lastgen_{layername}_chan{chan_id:03d}_{time_wdw_str}_{methodlab}_{RND}.jpg"))

#%%
sess.close()

