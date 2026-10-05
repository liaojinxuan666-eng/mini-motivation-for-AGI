import sys
import time
import torch
import torch.nn.functional as F

from model.world_model_conf import WorldModelConf
from data.gridworld_dynamics import make_world, make_sequence, VOCAB_SIZE


def compute_ece(confs, corrects, n_bins=10):
    """confs: [N] 概率, corrects: [N] 0/1"""
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = confs.numel()
    for i in range(n_bins):
        lo = bin_edges[i].item()
        hi = bin_edges[i + 1].item()
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if mask.sum().item() == 0:
            continue
        avg_conf = confs[mask].mean().item()
        avg_acc = corrects[mask].float().mean().item()
        ece += (mask.sum().item() / n) * abs(avg_conf - avg_acc)
    return ece


def evaluate(model, world, device, length=32, batch_size=128):
    model.eval()
    with torch.no_grad():
        seq, target = make_sequence(world, length, batch_size, device)
        coord_logits, conf_logit = model(seq, return_conf=True)

        pred = coord_logits[:, :-1].argmax(-1)
        tgt = target[:, 1:]
        conf = torch.sigmoid(conf_logit[:, :-1])

        mask = tgt != -100
        pred_m = pred[mask]
        tgt_m = tgt[mask]
        conf_m = conf[mask]

        correct = (pred_m == tgt_m).float()
        acc = correct.mean().item()
        ece = compute_ece(conf_m, correct)
    return acc, ece


def train_one(steps=2000, lr=3e-4, log_every=500, seed=42, device="cuda",
              lambda_conf=0.1):
    torch.manual_seed(seed)
    world = make_world(seed=seed)

    model = WorldModelConf(vocab_size=VOCAB_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train()
        seq, target = make_sequence(world, 32, 32, device)
        coord_logits, conf_logit = model(seq, return_conf=True)

        cl = coord_logits[:, :-1]
        cl_flat = cl.reshape(-1, VOCAB_SIZE)
        tg = target[:, 1:].reshape(-1)
        cf = conf_logit[:, :-1].reshape(-1)
        tg_for_conf = target[:, 1:]

        loss_coord = F.cross_entropy(cl_flat, tg, ignore_index=-100)

        with torch.no_grad():
            pred = cl.argmax(-1)
            valid = tg_for_conf != -100
            is_correct = (pred == tg_for_conf).float()

        mask_flat = valid.reshape(-1)
        loss_conf = F.binary_cross_entropy_with_logits(
            cf[mask_flat], is_correct.reshape(-1)[mask_flat]
        )

        loss = loss_coord + lambda_conf * loss_conf
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"step {step}/{steps}  total {avg:.4f}  "
                  f"coord {loss_coord.item():.4f}  conf {loss_conf.item():.4f}  "
                  f"elapsed {time.time()-t0:.0f}s")

    torch.save(model.state_dict(), "/kaggle/working/world_conf.pt")

    acc_same, ece_same = evaluate(model, make_world(seed=seed), device)
    acc_new, ece_new = evaluate(model, make_world(seed=999), device)

    print(f"\nparams={n_params:,}  time={time.time()-t0:.0f}s")
    print(f"训练世界: acc={acc_same:.4f}  ECE={ece_same:.4f}")
    print(f"全新世界: acc={acc_new:.4f}  ECE={ece_new:.4f}")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    train_one(steps=steps)