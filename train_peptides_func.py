import os
import sys
import time
import argparse
import logging
import numpy as np
import optax
import jax
from jax import random, jit, numpy as jnp, value_and_grad
from jax.tree_util import tree_leaves, tree_map
from flax.training import train_state
from model import Peptides, recurrent_param, no_decay_param
from datasets import load_peptides
from utils import map_nested_fn, eval_ap

max_nodes = 444
max_hops = 40

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=8, type=int)
parser.add_argument("--num_hops", default=40, type=int)
parser.add_argument("--dim_h", default=88, type=int)
parser.add_argument("--dim_v", default=88, type=int)
parser.add_argument("--r_min", default=0.95, type=float)
parser.add_argument("--r_max", default=1., type=float)
parser.add_argument("--max_phase", default=6.28, type=float)
parser.add_argument("--drop_rate", default=0.2, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)

#* training hyper-params
parser.add_argument("--lr_min", default=1e-7, type=float)
parser.add_argument("--lr_max", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=0.2, type=float)
parser.add_argument("--lr_factor", default=1., type=float)
parser.add_argument("--name", default="peptides-func", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--warmup", default=0.05, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
np.random.seed(args.seed)

if args.num_hops is None:
    args.num_hops = max_hops

if not os.path.exists("./log"):
    os.mkdir("./log")

time_str = time.strftime("%m%d-%H%M%S")
logging.basicConfig(filename=f"./log/{args.name}_{time_str}_{args.gpu}.log",
                    format="%(message)s", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(args)

class TrainState(train_state.TrainState):
    key: jax.Array
    train_loss: float
    eval_loss: float
    total: int
    logits: jax.Array

@jit
def train_step(state, batch):
    step_key = random.fold_in(state.key, state.step)
    def loss_fn(params):
        logits = state.apply_fn(params, batch["x"], batch["node_mask"], batch["dist_mask"], training=True, rngs={"dropout": step_key})
        loss = optax.sigmoid_binary_cross_entropy(
            logits=logits,
            labels=batch["y"]
        ).mean()
        return loss
    train_loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(train_loss = (state.train_loss + train_loss * batch["y"].shape[0]),
                          total = (state.total + batch["y"].shape[0]))
    return state

@jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch["x"], batch["node_mask"], batch["dist_mask"], training=False)
    eval_loss = optax.sigmoid_binary_cross_entropy(
        logits=logits,
        labels=batch["y"]
    ).mean()
    state = state.replace(eval_loss = (state.eval_loss + eval_loss * batch["y"].shape[0]),
                          total = (state.total + batch["y"].shape[0]),
                          logits = logits)
    return state

def main():
    train_set, val_set, test_set = load_peptides(args.name)
    model = Peptides(
        num_layers=args.num_layers,
        dim_o=10,
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
    dist_mask = np.zeros((args.batch_size, max_hops, max_nodes, max_nodes), dtype=np.bool_)
    for i in range(args.batch_size):
        dist_mask[i, :train_set[1][i].shape[0], :train_set[1][i].shape[1], :train_set[1][i].shape[2]] = train_set[1][i]
    params = model.init(params_key,
                        train_set[0]["x"][:args.batch_size],
                        train_set[0]["node_mask"][:args.batch_size],
                        dist_mask[:, :args.num_hops],
                        training=False)
    logging.info(f"# parameters: {sum(p.size for p in tree_leaves(params))}")
    
    # compute number of batches for train/val/test
    train_size = train_set[0]["x"].shape[0]
    train_steps_per_epoch = train_size // args.batch_size
    train_steps_total = train_steps_per_epoch * args.epochs

    val_size = val_set[0]["x"].shape[0]
    val_steps = (val_size - 1) // args.batch_size + 1

    test_size = test_set[0]["x"].shape[0]
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
                    warmup_steps=int(train_steps_total * args.warmup),
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "no_decay": optax.inject_hyperparams(optax.adam)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=int(train_steps_total * args.warmup),
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "regular": optax.inject_hyperparams(optax.adamw)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=int(train_steps_total * args.warmup),
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
        total=0,
        logits=None
    )


    best_val_ap = 0.
    ckpt = None
    ckpt_at = 0
    patience = 0
    for e in range(args.epochs):
        start = time.time()
        train_indices = np.random.permutation(train_size)
        for s in range(train_steps_per_epoch):
            batch_indices = train_indices[s * args.batch_size:(s + 1) * args.batch_size]
            dist_mask = np.zeros((len(batch_indices), max_hops, max_nodes, max_nodes), dtype=np.bool_)
            for i, idx in enumerate(batch_indices):
                dist_mask[i, :train_set[1][idx].shape[0], :train_set[1][idx].shape[1], :train_set[1][idx].shape[2]] = train_set[1][idx]
            batch = {
                "x": train_set[0]["x"][batch_indices],
                "y": train_set[0]["y"][batch_indices],
                "node_mask": train_set[0]["node_mask"][batch_indices],
                "dist_mask": dist_mask[:, :args.num_hops]
            }
            state = train_step(state, batch)
        
        logging.info(f"Epoch: {e + 1}; Training loss: {state.train_loss / state.total}")
        state = state.replace(train_loss=0., total=0)

        val_indices = np.arange(val_size)
        y_pred = []; y_true = []
        for s in range(val_steps):
            batch_indices = val_indices[s * args.batch_size:(s + 1) * args.batch_size]
            dist_mask = np.zeros((len(batch_indices), max_hops, max_nodes, max_nodes), dtype=np.bool_)
            for i, idx in enumerate(batch_indices):
                dist_mask[i, :val_set[1][idx].shape[0], :val_set[1][idx].shape[1], :val_set[1][idx].shape[2]] = val_set[1][idx]
            batch = {
                "x": val_set[0]["x"][batch_indices],
                "y": val_set[0]["y"][batch_indices],
                "node_mask": val_set[0]["node_mask"][batch_indices],
                "dist_mask": dist_mask[:, :args.num_hops]
            }
            state = eval_step(state, batch)
            y_true.append(batch["y"])
            y_pred.append(np.asarray(state.logits))
        
        val_loss = state.eval_loss / state.total
        val_ap = eval_ap(y_true=np.concatenate(y_true), y_pred=np.concatenate(y_pred))
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            ckpt_at = e + 1
            ckpt = state.params
            patience = 0
        else:
            patience = patience + 1
            if patience == 50:
                break
        
        logging.info(f"Epoch: {e + 1}; Val loss: {val_loss}; Val ap: {val_ap}")
        logging.info(f"Time per epoch: {time.time() - start} seconds")
        state = state.replace(eval_loss=0., total=0)
    
    state = state.replace(params=ckpt)
    test_indices = np.arange(test_size)
    y_pred = []; y_true = []
    for s in range(test_steps):
        batch_indices = test_indices[s * args.batch_size:(s + 1) * args.batch_size]
        dist_mask = np.zeros((len(batch_indices), max_hops, max_nodes, max_nodes), dtype=np.bool_)
        for i, idx in enumerate(batch_indices):
            dist_mask[i, :test_set[1][idx].shape[0], :test_set[1][idx].shape[1], :test_set[1][idx].shape[2]] = test_set[1][idx]
        batch = {
            "x": test_set[0]["x"][batch_indices],
            "y": test_set[0]["y"][batch_indices],
            "node_mask": test_set[0]["node_mask"][batch_indices],
            "dist_mask": dist_mask[:, :args.num_hops]
        }
        state = eval_step(state, batch)
        y_true.append(batch["y"])
        y_pred.append(np.asarray(state.logits))
    test_ap = eval_ap(y_true=np.concatenate(y_true), y_pred=np.concatenate(y_pred))
    logging.info(f"Best val ap {best_val_ap} at epoch {ckpt_at}; Test ap: {test_ap}")

if __name__ == "__main__":
    main()
