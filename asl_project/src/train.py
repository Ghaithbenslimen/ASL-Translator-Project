import copy, math
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from model import ASLNet
from dataset import make_loaders, CLASS_NAMES

CONFIG = dict(
    data_root="/content/dataset/asl_alphabet_train",
    out_dir="/content/drive/MyDrive/asl_project/checkpoints",
    img_size=128,
    batch_size=64,
    epochs=30,
    lr=1e-3,
    weight_decay=0.05,
    warmup_steps=500,
    grad_clip=5.0,
    ema_decay=0.999,
    num_workers=2,
    num_classes=len(CLASS_NAMES),
)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    train_loader, val_loader = make_loaders(
        CONFIG["data_root"], CONFIG["img_size"],
        CONFIG["batch_size"], CONFIG["num_workers"])
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = ASLNet(num_classes=CONFIG["num_classes"], width=0.5).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)

    def update_ema(d):
        with torch.no_grad():
            for e, m in zip(ema.parameters(), model.parameters()):
                e.mul_(d).add_(m, alpha=1-d)

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    total = CONFIG["epochs"] * len(train_loader)
    warm = CONFIG["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s/warm if s < warm else 0.5*(1+math.cos(math.pi*(s-warm)/(total-warm))))
    scaler = GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = 0.0

    for ep in range(CONFIG["epochs"]):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{CONFIG['epochs']}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with autocast(device_type="cuda"):
                logits = model(x)
                loss = crit(logits, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            update_ema(CONFIG["ema_decay"])
            tl += loss.item()*x.size(0)
            tc += (logits.argmax(1)==y).sum().item(); tt += x.size(0)
            pbar.set_postfix({"loss": f"{tl/tt:.4f}", "acc": f"{100*tc/tt:.2f}%"})

        ema.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="  val"):
                x, y = x.to(device), y.to(device)
                with autocast(device_type="cuda"):
                    logits = ema(x); loss = crit(logits, y)
                vl += loss.item()*x.size(0)
                vc += (logits.argmax(1)==y).sum().item(); vt += x.size(0)
        va = 100*vc/vt
        print(f"  → val_loss={vl/vt:.4f}  val_acc={va:.2f}%")
        if va > best:
            best = va
            torch.save({
                "ema_state": ema.state_dict(),
                "model_state": model.state_dict(),
                "config": CONFIG,
                "val_acc": va,
            }, f"{CONFIG['out_dir']}/best.pt")
            print(f"  ✓ saved (best={va:.2f}%)")
    print(f"\nDONE. Best val acc: {best:.2f}%")


if __name__ == "__main__":
    train()
