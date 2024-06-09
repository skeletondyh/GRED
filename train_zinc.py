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
from model import ZINC, recurrent_param, no_decay_param
from datasets import load_zinc
from utils import map_nested_fn

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=11, type=int)
parser.add_argument("--num_hops", default=5, type=int)
parser.add_argument("--dim_h", default=72, type=int)
parser.add_argument("--dim_v", default=72, type=int)
parser.add_argument("--r_min", default=0.9, type=float)
parser.add_argument("--r_max", default=1., type=float)
parser.add_argument("--max_phase", default=6.28, type=float)
parser.add_argument("--drop_rate", default=0.2, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)

#* training hyper-params
parser.add_argument("--lr_min", default=1e-7, type=float)
parser.add_argument("--lr_max", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=0.1, type=float)
parser.add_argument("--lr_factor", default=1., type=float)
parser.add_argument("--epochs", default=2000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--warmup", default=0.05, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
np.random.seed(args.seed)

if not os.path.exists("./log"):
    os.mkdir("./log")

time_str = time.strftime("%m%d-%H%M%S")
logging.basicConfig(filename=f"./log/ZINC_{time_str}_{args.gpu}.log",
                    format="%(message)s", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(args)

class TrainState(train_state.TrainState):
    key: random.KeyArray
    train_loss: float
    eval_loss: float
    total: int

@jit
def train_step(state, batch):
    step_key = random.fold_in(state.key, state.step)
    def loss_fn(params):
        logits = state.apply_fn(params, batch["x"], batch["node_mask"], batch["dist_mask"], batch["edge_attr"], training=True, rngs={"dropout": step_key})
        loss = jnp.abs(logits.squeeze() - batch["y"]).mean()
        return loss
    train_loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(train_loss = (state.train_loss + train_loss * batch["y"].shape[0]),
                          total = (state.total + batch["y"].shape[0]))
    return state

@jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch["x"], batch["node_mask"], batch["dist_mask"], batch["edge_attr"], training=False)
    eval_loss = jnp.abs(logits.squeeze() - batch["y"]).sum()
    state = state.replace(eval_loss = (state.eval_loss + eval_loss),
                          total = (state.total + batch["y"].shape[0]))
    return state

def main():
    train_set, val_set, test_set = load_zinc()
    model = ZINC(
        num_layers=args.num_layers,
        dim_o=1,
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
                        train_set["edge_attr"][:args.batch_size],
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
        total=0
    )

    best_val_mae = 100.
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
                "dist_mask": train_set["dist_mask"][batch_indices, :args.num_hops],
                "edge_attr": train_set["edge_attr"][batch_indices]
            }
            state = train_step(state, batch)
        
        logging.info(f"Epoch: {e + 1}; Training loss: {state.train_loss / state.total}")
        state = state.replace(train_loss=0., total=0)

        val_indices = np.arange(val_size)
        for s in range(val_steps):
            batch_indices = val_indices[s * args.batch_size:(s + 1) * args.batch_size]
            batch = {
                "x": val_set["x"][batch_indices],
                "y": val_set["y"][batch_indices],
                "node_mask": val_set["node_mask"][batch_indices],
                "dist_mask": val_set["dist_mask"][batch_indices, :args.num_hops],
                "edge_attr": val_set["edge_attr"][batch_indices]
            }
            state = eval_step(state, batch)
        
        val_loss = state.eval_loss / state.total
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            ckpt_at = e + 1
            ckpt = state.params
        
        logging.info(f"Epoch: {e + 1}; Val loss: {val_loss}")
        logging.info(f"Time per epoch: {time.time() - start} seconds")
        state = state.replace(eval_loss=0., total=0)
    
    state = state.replace(params=ckpt)
    test_indices = np.arange(test_size)
    for s in range(test_steps):
        batch_indices = test_indices[s * args.batch_size:(s + 1) * args.batch_size]
        batch = {
            "x": test_set["x"][batch_indices],
            "y": test_set["y"][batch_indices],
            "node_mask": test_set["node_mask"][batch_indices],
            "dist_mask": test_set["dist_mask"][batch_indices, :args.num_hops],
            "edge_attr": test_set["edge_attr"][batch_indices]
        }
        state = eval_step(state, batch)
    
    logging.info(f"Best val mae {best_val_mae} at epoch {ckpt_at}; Test mae: {state.eval_loss / state.total}")

if __name__ == "__main__":
    main()
