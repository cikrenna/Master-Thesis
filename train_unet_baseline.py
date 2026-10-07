# %% ── Training U-Net baseline (v3) — label da catalogo FoF/SubFind ──────────
"""
Secondo le indicazioni del prof. Benitez-Llambay (incontro 29/09):
  "pick a fast architecture just to see if everything works" → U-Net 2D minimale,
  input (1, 256, 256) densità SPH, output (1, 256, 256) sigmoid = maschera di halo.

NOTA metodologica (da citare in tesi): la verità-terra è il catalogo FoF+SubFind.
L'obiettivo NON è battere FoF+unbinding (impossibile: la rete vede solo una proiezione
2D e non ha l'informazione 3D sulle energie di legame), ma produrre un catalogo
equivalente nel modo più veloce possibile (completezza + efficienza).

Dataset atteso (dal notebook CAMELS_preprocessing v3.0, cartella preprocessed_v3.0):
  X_patches.npy           (N, 1, 256, 256) float32  — densità SPH normalizzata (1 canale)
  Y_mask_patches.npy      (N, 256, 256)   uint8    — label primaria (maschera 0/1 dal catalogo)
  Y_heatmap_patches.npy / Y_radius_patches.npy     — per lo stadio 2 (CenterNet/StarDist-like),
                                                      non usati in questa baseline.
  norm_stats.npy          (1, 2) float32  — [MED_D, SIG_D] (ridondante: X è già normalizzato)

ATTENZIONE overfitting: con UNA sola simulazione (400 patch adiacenti correlate) i numeri
sono indicativi ma ottimistici. Soluzione indicata dal prof: passare a 5–10 simulazioni
CAMELS (≈2.000–4.000 patch); probabilmente NON serve data augmentation ("plenty of data").

Esempio d'uso (GPU consigliata; su CPU usare --epochs ridotte e SUBSET):
  python train_unet_baseline.py --data_dir ".../preprocessed_v3.0" --epochs 50
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ══════════════════════════════════════════════════════════════════════════════
# 1) U-Net 2D minimal (~1M parametri): encoder-decoder con skip connections
# ══════════════════════════════════════════════════════════════════════════════
class DoubleConv(nn.Module):
    """Blocco: (conv3x3 → BN → ReLU) × 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net classica (Ronneberger 2015), base 64: ~1.75M parametri.
    input (B,1,256,256) → logits (B,1,256,256); sigmoide applicata fuori (BCEWithLogits)."""

    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        ch = [base * (2 ** i) for i in range(4)]          # 64,128,256,512
        self.enc1 = DoubleConv(in_ch, ch[0])
        self.enc2 = DoubleConv(ch[0], ch[1])
        self.enc3 = DoubleConv(ch[1], ch[2])
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(ch[2], ch[3])
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3 = DoubleConv(ch[2] * 2, ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = DoubleConv(ch[1] * 2, ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = DoubleConv(ch[0] * 2, ch[0])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ══════════════════════════════════════════════════════════════════════════════
# 2) Loss: BCE + Dice (lo standard della segmentazione aiuta con classi sbilanciate:
#    i pixel di alone sono una piccola frazione dell'immagine)
# ══════════════════════════════════════════════════════════════════════════════
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def combined_loss(logits, target, w_dice=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + w_dice * dice_loss(logits, target)


# ══════════════════════════════════════════════════════════════════════════════
# 3) Metriche: IoU / Dice coefficiente (su soglia 0.5)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def iou_dice(model, loader, device):
    model.eval()
    inter_sum = pred_sum = true_sum = 0.0
    n_img = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if y.dim() == 3:
            y = y[:, None]
        prob = torch.sigmoid(model(x))
        pred = (prob >= 0.5).float()
        inter_sum += (pred * y).sum().item()
        pred_sum += pred.sum().item()
        true_sum += y.sum().item()
        n_img += x.shape[0]
    iou = inter_sum / max(pred_sum + true_sum - inter_sum, 1e-9)
    dice = 2.0 * inter_sum / max(pred_sum + true_sum, 1e-9)
    return iou, dice, n_img


# ══════════════════════════════════════════════════════════════════════════════
# 4) Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="U-Net baseline su patch CAMELS v3")
    ap.add_argument("--data_dir", default=r"C:\Users\kekko\Downloads\Python\TESI\CAMELS_data\preprocessed_v3.0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=256, help="lato patch (256 in produzione; 64 per smoke test)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="usa solo N patch (debug CPU)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Caricamento dataset ────────────────────────────────────────────────
    X = np.load(os.path.join(args.data_dir, "X_patches.npy"))            # (N,1,S,S) f32
    X = X[..., :args.img_size, :args.img_size]   # crop per smoke test su piccoli tensori
    Y = np.load(os.path.join(args.data_dir, "Y_mask_patches.npy"))       # (N,S,S) u8
    Y = Y[..., :args.img_size, :args.img_size]
    assert X.ndim == 4 and X.shape[1] == 1, \
        f"Atteso X (N,1,256,256) da v3; ottenuto {X.shape}. Rigenerare col notebook v3.0."
    if args.subset:
        X, Y = X[:args.subset], Y[:args.subset]

    # Y è (N,S,S): aggiunge la dimensione canale per allinearsi ai logits (N,1,S,S)
    y_t = torch.from_numpy(Y.astype(np.float32))[:, None, :, :]
    ds = TensorDataset(torch.from_numpy(X), y_t)
    n_val = max(int(0.15 * len(ds)), 1)
    n_test = max(int(0.15 * len(ds)), 1)
    n_train = len(ds) - n_val - n_test
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)
    print(f"Patch totali {len(ds)} | train {n_train} | val {n_val} | test {n_test}")
    print("⚠️  Split CASUALE su patch della STESSA simulazione/fette adiacenti: "
          "le metriche sono ottimistiche finché non si usano ≥5 simulazioni.")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── Modello / ottimizzatore ────────────────────────────────────────────
    model = UNet(in_ch=1, out_ch=1, base=64).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"U-Net parametri: {n_par:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_iou = -1.0
    ckpt_dir = os.path.join(args.data_dir, "unet_baseline_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = combined_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.shape[0]
        sched.step()
        avg = run_loss / n_train
        val_iou, val_dice, _ = iou_dice(model, val_dl, device)
        print(f"epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  val IoU={val_iou:.4f}  val Dice={val_dice:.4f}")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_iou": val_iou}, os.path.join(ckpt_dir, "best_unet.pt"))

    # ── Test finale con il miglior checkpoint ──────────────────────────────
    ck = torch.load(os.path.join(ckpt_dir, "best_unet.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    test_iou, test_dice, n_test_img = iou_dice(model, test_dl, device)
    print("\n=== Risultati sul set di test ===")
    print(f"  patch test : {n_test_img}")
    print(f"  IoU        : {test_iou:.4f}")
    print(f"  Dice       : {test_dice:.4f}")
    print("Prossimo passo (§7.3 handoff): post-processing connectivity "
          "(scipy.ndimage.label) → catalogo oggetti → completeness/purity vs FoF.")


if __name__ == "__main__":
    main()
