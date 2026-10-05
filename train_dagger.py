import sys
import time
import torch
import torch.nn.functional as F

from model.world_model_conf import WorldModelConf
from data.gridworld_dynamics import make_world, make_sequence, VOCAB_SIZE


def perturb_sequence(seq, pred, p=0.3):
    """在坐标位置（3,4,6,7,9,10,...）随机用模型预测替换真实 token。"""
    B, L = seq.shape
    seq_p = seq.clone()
    coord_pos = [i for i in range(3, L) if i % 3 in (0, 1)]
    coord_pos_t = torch.tensor(coord_pos, device=seq.device)
    rand = torch.rand(B, len(coord_pos), device=seq.device)
    mask = rand < p
    pred_at = pred[:, coord_pos_t]
    seq_p[:, coord_pos_t] = torch.where(mask, pred_at, seq[:, coord_pos_t])
    return seq_p


def train_one(steps=2000, lr=3e-4, log_every=500, seed=42, device="cuda",
              lambda_conf=0.1, dagger_p=0.3, dagger_prob=0.5):
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

        if torch.rand(1).item() < dagger_prob:
            with torch.no_grad():
                pred = model(seq).argmax(-1)
            seq_in = perturb_sequence(seq, pred, p=dagger_p)
        else:
            seq_in = seq

        coord_logits, conf_logit = model(seq_in, return_conf=True)

        cl = coord_logits[:, :-1]
        tg = target[:, 1:]
        cl_flat = cl.reshape(-1, VOCAB_SIZE)
        tg_flat = tg.reshape(-1)
        cf_flat = conf_logit[:, :-1].reshape(-1)

        loss_coord = F.cross_entropy(cl_flat, tg_flat, ignore_index=-100)

        with torch.no_grad():
            pred_at = cl.argmax(-1)
            valid = tg != -100
            is_correct = (pred_at == tg).float()

        valid_flat = valid.reshape(-1)
        loss_conf = F.binary_cross_entropy_with_logits(
            cf_flat[valid_flat], is_correct.reshape(-1)[valid_flat]
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

    torch.save(model.state_dict(), "/kaggle/working/world_dagger.pt")
    print(f"\nparams={n_params:,}  time={time.time()-t0:.0f}s")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    train_one(steps=steps)