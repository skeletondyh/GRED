import os
import sys
import time
import argparse
import logging
import numpy as np
import optax
from jax import random, jit, numpy as jnp, value_and_grad
from jax.tree_util import tree_leaves, tree_map
from flax.training import train_state
from model import SuperPixel, recurrent_param, no_decay_param
from datasets import load_superpixel
from utils import map_nested_fn

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=8, type=int)
parser.add_argument("--num_hops", default=5, type=int)
parser.add_argument("--dim_h", default=96, type=int)
parser.add_argument("--dim_v", default=64, type=int)
parser.add_argument("--r_min", default=0., type=float)
parser.add_argument("--r_max", default=1., type=float)
parser.add_argument("--max_phase", default=6.28, type=float)
parser.add_argument("--drop_rate", default=0.15, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)

#* training hyper-params
parser.add_argument("--lr_min", default=1e-7, type=float)
parser.add_argument("--lr_max", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=0.1, type=float)
parser.add_argument("--lr_factor", default=1., type=float)
parser.add_argument("--name", default="CIFAR10", type=str)
parser.add_argument("--epochs", default=600, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
np.random.seed(args.seed)

if not os.path.exists("./log"):
    os.mkdir("./log")

time_str = time.strftime("%m%d-%H%M%S")
logging.basicConfig(filename=f"./log/{args.name}_{time_str}_{args.gpu}.log",
                    format="%(message)s", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(args)

class TrainState(train_state.TrainState):
    key: random.KeyArray
    train_loss: float
    eval_loss: float
    correct: int
    total: int

@jit
def train_step(state, batch):
    step_key = random.fold_in(state.key, state.step)
    def loss_fn(params):
        logits = state.apply_fn(params, batch["x"], batch["node_mask"], batch["dist_mask"], training=True, rngs={"dropout": step_key})
        correct = jnp.equal(logits.argmax(axis=-1), batch["y"]).sum()
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["y"]
        ).mean()
        return loss, correct
    (train_loss, correct), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(train_loss = (state.train_loss + train_loss * batch["y"].shape[0]),
                          correct = (state.correct + correct),
                          total = (state.total + batch["y"].shape[0]))
    return state

@jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch["x"], batch["node_mask"], batch["dist_mask"], training=False)
    eval_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch["y"]
    ).sum()
    correct = jnp.equal(logits.argmax(axis=-1), batch["y"]).sum()
    state = state.replace(eval_loss = (state.eval_loss + eval_loss),
                          correct = (state.correct + correct),
                          total = (state.total + batch["y"].shape[0]))
    return state

def main():
    train_set, val_set, test_set = load_superpixel(args.name)
    model = SuperPixel(
        num_layers=args.num_layers,
        dim_o=(train_set["y"].max() + 1),
        dim_v=args.dim_v,
        dim_h=args.dim_h,
        expand=args.expand,
        r_min=args.r_min,
        r_max=args.r_max,
        max_phase=args.max_phase,
        drop_rate=args.drop_rate,
        act=args.act
    )
    root_key = random.PRNGKey(args.seed)
    key, params_key, dropout_key = random.split(root_key, 3)
    params = model.init(params_key,
                        train_set["x"][:args.batch_size],
                        train_set["node_mask"][:args.batch_size],
                        train_set["dist_mask"][:args.batch_size, :args.num_hops],
                        training=False)
    logging.info(f"# parameters: {sum(p.size for p in tree_leaves(params))}")
    
    # compute number of batches for train/val/test
    train_size = train_set["x"].shape[0]
    train_steps_per_epoch = train_size // args.batch_size
    train_steps_total = train_steps_per_epoch * args.epochs

    val_size = val_set["x"].shape[0]
    val_steps = (val_size - 1) // args.batch_size + 1

    test_size = test_set["x"].shape[0]
    test_steps = (test_size - 1) // args.batch_size + 1
    
    logging.info(f"train size: {train_size}; # train steps per epoch: {train_steps_per_epoch}; # train steps total: {train_steps_total}")
    logging.info(f"val size: {val_size}; # val steps: {val_steps}")
    logging.info(f"test size: {test_size}; # test steps: {test_steps}")

    label_fn = map_nested_fn(
        lambda k, _: "recurrent" if k in recurrent_param else "no_decay" if k in no_decay_param else "regular"
    )
    tx = optax.multi_transform(
        {
            "recurrent": optax.inject_hyperparams(optax.adam)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max * args.lr_factor,
                    warmup_steps=train_steps_total // 20,
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "no_decay": optax.inject_hyperparams(optax.adam)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=train_steps_total // 20,
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "regular": optax.inject_hyperparams(optax.adamw)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=train_steps_total // 20,
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                ),
                weight_decay=args.weight_decay
            )
        },
        label_fn
    )

    state = TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=tx,
        key=dropout_key, 
        train_loss=0.,
        eval_loss=0.,
        correct=0,
        total=0
    )

    best_val_acc = 0.
    ckpt = None
    ckpt_at = 0
    for e in range(args.epochs):
        start = time.time()
        train_indices = np.random.permutation(train_size)
        for s in range(train_steps_per_epoch):
            batch_indices = train_indices[s * args.batch_size:(s + 1) * args.batch_size]
            batch = {
                "x": train_set["x"][batch_indices],
                "y": train_set["y"][batch_indices],
                "node_mask": train_set["node_mask"][batch_indices],
                "dist_mask": train_set["dist_mask"][batch_indices, :args.num_hops]
            }
            state = train_step(state, batch)
        
        logging.info(f"Epoch: {e + 1}; Training loss: {state.train_loss / state.total}; Training acc: {state.correct / state.total}")
        state = state.replace(train_loss=0., correct=0, total=0)

        val_indices = np.arange(val_size)
        for s in range(val_steps):
            batch_indices = val_indices[s * args.batch_size:(s + 1) * args.batch_size]
            batch = {
                "x": val_set["x"][batch_indices],
                "y": val_set["y"][batch_indices],
                "node_mask": val_set["node_mask"][batch_indices],
                "dist_mask": val_set["dist_mask"][batch_indices, :args.num_hops],
            }
            state = eval_step(state, batch)
        
        val_loss = state.eval_loss / state.total
        val_acc = state.correct / state.total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_at = e + 1
            ckpt = state.params
        
        logging.info(f"Epoch: {e + 1}; Val loss: {val_loss}; Val acc: {val_acc}")
        logging.info(f"Time per epoch: {time.time() - start} seconds")
        state = state.replace(eval_loss=0., correct=0, total=0)
    
    state = state.replace(params=ckpt)
    test_indices = np.arange(test_size)
    for s in range(test_steps):
        batch_indices = test_indices[s * args.batch_size:(s + 1) * args.batch_size]
        batch = {
            "x": test_set["x"][batch_indices],
            "y": test_set["y"][batch_indices],
            "node_mask": test_set["node_mask"][batch_indices],
            "dist_mask": test_set["dist_mask"][batch_indices, :args.num_hops],
        }
        state = eval_step(state, batch)
    
    logging.info(f"Best val acc {best_val_acc} at epoch {ckpt_at}; Test acc: {state.correct / state.total}")

if __name__ == "__main__":
    main()
