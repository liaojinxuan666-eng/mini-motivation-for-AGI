import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import torch
import torch.nn.functional as F

from model.world_model_conf import WorldModelConf
from data.gridworld_dynamics import make_world, make_sequence, VOCAB_SIZE


def train_one(steps=3000, lr=3e-4, log_every=100, seed=42, device="cuda",
              rollout_steps=3, seq_len=32, batch_size=8,
              warmup_frac=0.5):
    torch.manual_seed(seed)
    world = make_world(seed=seed)
    model = WorldModelConf(vocab_size=VOCAB_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    warmup_steps = int(steps * warmup_frac)
    t0 = time.time()
    losses = []

    for step in range(1, steps + 1):
        model.train()
        seq, target = make_sequence(world, seq_len, batch_size, device)

        if step <= warmup_steps:
            # 阶段 1：教师强制，正常 next-token 预测
            logits = model(seq)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                target[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            mode = "TF"
        else:
            # 阶段 2：rollout 训练，只在最后 rollout_steps 步算 loss
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
            mode = "RO"

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % 50 == 0:
            torch.cuda.empty_cache()

        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"step {step}/{steps}  [{mode}] loss {avg:.4f}  "
                  f"elapsed {time.time()-t0:.0f}s")

    torch.save(model.state_dict(), "/kaggle/working/world_rollout.pt")
    print(f"\ntime={time.time()-t0:.0f}s")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    train_one(steps=steps)