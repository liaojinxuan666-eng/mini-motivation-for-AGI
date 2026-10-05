import sys
import time
import torch
import torch.nn.functional as F

from model.world_model_conf import WorldModelConf
from data.gridworld_dynamics import make_world, make_sequence, VOCAB_SIZE


def train_one(steps=3000, lr=3e-4, log_every=500, seed=42, device="cuda",
              rollout_steps=3, seq_len=32, batch_size=4):
    torch.manual_seed(seed)
    world = make_world(seed=seed)
    model = WorldModelConf(vocab_size=VOCAB_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train()
        seq, _ = make_sequence(world, seq_len, batch_size, device)

        prefix_len = seq_len - rollout_steps
        prefix = seq[:, :prefix_len * 3]
        total_loss = 0.0

        cur = prefix
        for t in range(rollout_steps):
            pos = prefix_len + t
            gt_x = seq[:, pos * 3]
            gt_y = seq[:, pos * 3 + 1]
            gt_a = seq[:, pos * 3 + 2]

            logits = model(cur)
            loss_x = F.cross_entropy(logits[:, -1], gt_x)
            pred_x = logits[:, -1].argmax(-1)
            cur = torch.cat([cur, pred_x.unsqueeze(1)], dim=1)

            logits = model(cur)
            loss_y = F.cross_entropy(logits[:, -1], gt_y)
            pred_y = logits[:, -1].argmax(-1)
            cur = torch.cat([cur, pred_y.unsqueeze(1)], dim=1)

            cur = torch.cat([cur, gt_a.unsqueeze(1)], dim=1)

            total_loss = total_loss + (loss_x + loss_y)

        loss = total_loss / rollout_steps
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % 100 == 0:
            torch.cuda.empty_cache()

        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"step {step}/{steps}  loss {avg:.4f}  elapsed {time.time()-t0:.0f}s")

    torch.save(model.state_dict(), "/kaggle/working/world_rollout.pt")
    print(f"\ntime={time.time()-t0:.0f}s")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    train_one(steps=steps)