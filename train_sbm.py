import os
import sys
import time
import argparse
import logging
import numpy as np
import optax
from jax import Array, random, jit, numpy as jnp, value_and_grad
from jax.tree_util import tree_leaves, tree_map
from flax.training import train_state
from model import SBM, recurrent_param, no_decay_param
from datasets import load_sbm
from utils import map_nested_fn

parser = argparse.ArgumentParser()
#* model hyper-params
parser.add_argument("--num_layers", default=16, type=int)
parser.add_argument("--num_hops", default=None, type=int)
parser.add_argument("--dim_h", default=64, type=int)
parser.add_argument("--dim_v", default=64, type=int)
parser.add_argument("--r_min", default=0.9, type=float)
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
parser.add_argument("--name", default="CLUSTER", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=32, type=int)
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
    correct: Array
    total: Array

@jit
def train_step(state, batch):
    step_key = random.fold_in(state.key, state.step)
    labels = jnp.where(batch["node_mask"], batch["y"], args.num_cls)
    cls_freq = jnp.bincount(labels.reshape(-1), length=args.num_cls) / batch["node_mask"].sum() 
    weights = jnp.zeros(batch["y"].shape)
    for i in range(args.num_cls):
        weights = jnp.where(labels == i, 1 - cls_freq[i], weights)
    def loss_fn(params):
        logits = state.apply_fn(params, batch["x"], batch["dist_mask"], training=True, rngs={"dropout": step_key})
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["y"]
        )
        loss = (weights * cross_entropy).sum() / weights.sum()
        return loss
    train_loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(train_loss=train_loss)
    return state

@jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch["x"], batch["dist_mask"], training=False)
    labels = jnp.where(batch["node_mask"], batch["y"], args.num_cls)
    pred = logits.argmax(axis=-1)
    correct = jnp.asarray([jnp.logical_and(labels == i, pred == i).sum() for i in range(args.num_cls)])
    total = jnp.asarray([(labels == i).sum() for i in range(args.num_cls)])
    state = state.replace(correct = (state.correct + correct),
                          total = (state.total + total))
    return state

def main():
    train_set, val_set, test_set = load_sbm(args.name)
    args.num_cls = train_set["y"].max() + 1
    model = SBM(
        num_layers=args.num_layers,
        dim_o=args.num_cls,
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
                        train_set["dist_mask"][:args.batch_size],
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
                    warmup_steps=train_steps_total // 40,
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "no_decay": optax.inject_hyperparams(optax.adam)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=train_steps_total // 40,
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "regular": optax.inject_hyperparams(optax.adamw)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=train_steps_total // 40,
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
        correct=jnp.zeros((args.num_cls,)),
        total=jnp.zeros((args.num_cls,))
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
                "dist_mask": train_set["dist_mask"][batch_indices]
            }
            state = train_step(state, batch)
        
        logging.info(f"Epoch: {e + 1}; Training loss: {state.train_loss}")
        state = state.replace(train_loss=0., correct=jnp.zeros((args.num_cls,)), total=jnp.zeros((args.num_cls,)))

        val_indices = np.arange(val_size)
        for s in range(val_steps):
            batch_indices = val_indices[s * args.batch_size:(s + 1) * args.batch_size]
            batch = {
                "x": val_set["x"][batch_indices],
                "y": val_set["y"][batch_indices],
                "node_mask": val_set["node_mask"][batch_indices],
                "dist_mask": val_set["dist_mask"][batch_indices]
            }
            state = eval_step(state, batch)
        
        val_acc = (state.correct / state.total).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_at = e + 1
            ckpt = state.params
        
        logging.info(f"Epoch: {e + 1}; Val acc: {val_acc}")
        logging.info(f"Time per epoch: {time.time() - start} seconds")
        state = state.replace(correct=jnp.zeros((args.num_cls,)), total=jnp.zeros((args.num_cls,)))
    
    state = state.replace(params=ckpt)
    test_indices = np.arange(test_size)
    for s in range(test_steps):
        batch_indices = test_indices[s * args.batch_size:(s + 1) * args.batch_size]
        batch = {
            "x": test_set["x"][batch_indices],
            "y": test_set["y"][batch_indices],
            "node_mask": test_set["node_mask"][batch_indices],
            "dist_mask": test_set["dist_mask"][batch_indices]
        }
        state = eval_step(state, batch)
    
    logging.info(f"Best val acc {best_val_acc} at epoch {ckpt_at}; Test acc: {(state.correct / state.total).mean()}")

if __name__ == "__main__":
    main()
