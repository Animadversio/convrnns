import os
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
from convrnns.utils.loader import get_restore_vars, MODEL_TO_KWARGS
from convrnns.models.model_func import model_func


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
from easydict import EasyDict as edict
args = edict({
    "model_name": 'rgc_intermediate',  # Required argument, so no default value is effectively used
    "out_layers": 'conv9',
    "gpu": None,
    "ckpt_dir": "./ckpts", #"/home/biw905/Github/convrnns/ckpts",#
    "image_pres": "default",
    "times": None,
    "image_off": None,
    "include_all_times": False,
    "include_logits": False
})

out_layers = args.out_layers.split(",")
model_kwargs = MODEL_TO_KWARGS[args.model_name]

# x would be the actual images you want to run through the model
num_imgs = 20
x = np.zeros((num_imgs, 224, 224, 3))
x = normalize_ims(x)
# new_graph = tf.Graph()
# with new_graph.as_default():
inputs = tf.placeholder(tf.float32, shape=[num_imgs, 224, 224, 3])
# with tf.variable_scope('scope_name', reuse=True):
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
CKPT_PATH = os.path.join(args.ckpt_dir, "{}/model.ckpt".format(args.model_name))
restore_vars = get_restore_vars(CKPT_PATH)
tf.train.Saver(var_list=restore_vars).restore(sess, CKPT_PATH)
y_eval = sess.run(y, feed_dict={inputs: x})
# repeat using these ...
print_keys_recursive(y_eval)
sess.close()

