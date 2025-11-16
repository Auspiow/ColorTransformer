# siamese_color_train.py
# pip install numpy pandas matplotlib seaborn scipy torch torchvision colormath colour-science

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fix Windows OpenMP crash
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import json, time

# color conversions
from colormath.color_objects import XYZColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# compat for old colormath calling numpy.asscalar
if not hasattr(np, "asscalar"):
    np.asscalar = lambda x: x.item()

plt.rcParams['figure.figsize'] = (7,5)
sns.set_theme(style="whitegrid")

# ------------------------
# 0) Config
# ------------------------
DATASET_DIR = "datasets"    # your folder with json files
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ------------------------
# 1) load JSON datasets -> DataFrame with L1,a1,b1,L2,a2,b2,DE_human
#    (works with rit-dupont / other jsons matching that format)
# ------------------------
def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    dv = np.array(d.get("dv", []))
    pairs = np.array(d.get("pairs", []), dtype=int)
    xyz = np.array(d.get("xyz", []), dtype=float)
    if xyz.size == 0 or pairs.size == 0 or dv.size == 0:
        return pd.DataFrame(columns=["L1","a1","b1","L2","a2","b2","DE_human"])
    # convert xyz -> lab (assume d65 & 2deg observer as earlier)
    lab_list = []
    for (x,y,z) in xyz:
        xyz_obj = XYZColor(x, y, z, observer='2', illuminant='d65')
        lab = convert_color(xyz_obj, LabColor)
        lab_list.append([lab.lab_l, lab.lab_a, lab.lab_b])
    lab_arr = np.array(lab_list)
    rows = []
    # zip pairs and dv; some files have dv length equals pairs length
    n = min(len(pairs), len(dv))
    for idx in range(n):
        i,j = pairs[idx]
        score = float(dv[idx])
        L1,a1,b1 = lab_arr[i]
        L2,a2,b2 = lab_arr[j]
        rows.append([L1,a1,b1,L2,a2,b2,score])
    return pd.DataFrame(rows, columns=["L1","a1","b1","L2","a2","b2","DE_human"])

# scan datasets dir
dfs = []
for fname in os.listdir(DATASET_DIR):
    if fname.lower().endswith(".json"):
        full = os.path.join(DATASET_DIR, fname)
        print("Loading", full)
        try:
            df = load_json_dataset(full)
            if len(df):
                dfs.append(df)
        except Exception as e:
            print("Failed to read", full, e)

if not dfs:
    raise RuntimeError("No JSON datasets found or parsed in 'datasets' directory.")

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.dropna().reset_index(drop=True)
print("Total pairs loaded:", len(df_all))

# ------------------------
# quick EDA: distribution of DE_human
# ------------------------
print(df_all["DE_human"].describe())
plt.figure()
sns.histplot(df_all["DE_human"].values, bins=60, kde=False)
plt.title("Human ΔE distribution (raw)")
plt.savefig("hist_DE_human_raw.png", dpi=150)
plt.close()

# ------------------------
# 2) Preprocess and augmentation:
#    - swap augmentation (L1<->L2)
#    - target: we will train on log1p(DE) to reduce imbalance
# ------------------------
X = df_all[["L1","a1","b1","L2","a2","b2"]].values.astype(np.float32)
y_raw = df_all["DE_human"].values.astype(np.float32).reshape(-1,1)

# swap augmentation to enforce symmetry
X_swap = X.copy()
X_swap[:, :3], X_swap[:, 3:] = X[:, 3:].copy(), X[:, :3].copy()
y_swap = y_raw.copy()

X = np.concatenate([X, X_swap], axis=0)
y_raw = np.concatenate([y_raw, y_swap], axis=0)

# log1p transform target
y_log = np.log1p(y_raw)  # shape (N,1)

# normalization (mean/std) on log-space
y_mean, y_std = y_log.mean(), y_log.std()
y_norm = (y_log - y_mean) / (y_std + 1e-9)

print("After augmentation total samples:", len(y_norm))

# ------------------------
# 3) Balanced sampling
#    - bin raw DE into K bins
#    - compute inverse-frequency weights per sample
# ------------------------
num_bins = 10
bins = np.linspace(0.0, max( y_raw.max(), 1.0 ), num_bins+1)
bin_idx = np.digitize(y_raw.ravel(), bins) - 1   # 0..num_bins-1
# avoid empty bins by clipping
bin_idx = np.clip(bin_idx, 0, num_bins-1)
counts = np.bincount(bin_idx, minlength=num_bins).astype(float)
# sample weight = 1 / count(bin)
weights = 1.0 / (counts[bin_idx] + 1e-9)
# normalize weights (optional)
weights = weights * (len(weights) / weights.sum())

# create torch tensors
X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y_norm.astype(np.float32)).squeeze(-1)

dataset = TensorDataset(X_t, y_t)
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
# We'll use sampler for training loader; validation uses simple split
# make indices for train/val
indices = list(range(len(dataset)))
np.random.shuffle(indices)
train_idx, val_idx = indices[:train_size], indices[train_size:]

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)  # we'll override with WeightedRandomSampler via DataLoader argument
# But DataLoader accepts sampler or shuffle; easiest: build train_loader using WeightedRandomSampler on full dataset
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=WeightedRandomSampler(weights[train_idx], num_samples=len(train_idx), replacement=True))
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_idx), "Val samples:", len(val_idx))

# ------------------------
# 4) Model: Siamese MLP encoder + distance MLP
# ------------------------
class SiameseColorNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        # encoder: maps (L,a,b) -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU()
        )
        # head: takes absolute difference |e1-e2| -> predict scalar (log1p(DE) normalized)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)   # output normalized log(DE)
        )

    def forward(self, x):
        # x shape (B,6)
        B = x.shape[0]
        colors = x.view(B, 2, 3)      # (B,2,3)
        c1 = colors[:,0,:]
        c2 = colors[:,1,:]
        e1 = self.encoder(c1)
        e2 = self.encoder(c2)
        d = torch.abs(e1 - e2)
        out = self.head(d).squeeze(-1)
        return out

model = SiameseColorNet(emb_dim=128).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
loss_fn = nn.HuberLoss(delta=1.0)   # robust to outliers

# ------------------------
# 5) Training loop
# ------------------------
best_val = 1e9
save_path = "checkpoints_siamese.pth"
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    n_seen = 0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE).float()
        yb = yb.to(DEVICE).float()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
        n_seen += xb.size(0)
    train_loss = running_loss / (n_seen + 1e-9)

    # validation
    model.eval()
    vloss = 0.0
    vcount = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            pred = model(xb)
            vloss += ((pred - yb)**2).sum().item()
            vcount += xb.size(0)
    val_mse = vloss / (vcount + 1e-9)

    print(f"Epoch {epoch:02d}  train_loss={train_loss:.6f}  val_mse={val_mse:.6f}  time={time.time()-t0:.1f}s")
    if val_mse < best_val:
        best_val = val_mse
        torch.save(model.state_dict(), save_path)
        print(" Saved best model.")

# ------------------------
# 6) Evaluate on full dataset: compute Pearson R vs human (raw DE)
# ------------------------
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model.eval()

with torch.no_grad():
    X_tensor = X_t.to(DEVICE).float()
    preds_norm = model(X_tensor).cpu().numpy().reshape(-1,1)   # normalized log-space
# unnormalize
preds_log = preds_norm * (y_std + 1e-9) + y_mean
preds_un = np.expm1(preds_log)   # back to raw DE

true_all = y_raw.reshape(-1,1)

# compute baseline ΔE2000 (may be slow)
def de2000_array_from_df(Xin, raw_y=None):
    # Xin shape Nx6 (L1,a1,b1,L2,a2,b2)
    out = []
    for row in Xin:
        c1 = LabColor(row[0], row[1], row[2])
        c2 = LabColor(row[3], row[4], row[5])
        val = delta_e_cie2000(c1, c2)
        # ensure python float
        if hasattr(val, "item"):
            val = val.item()
        out.append(float(val))
    return np.array(out).reshape(-1,1)

print("Computing ΔE2000 baseline (this may take time)...")
de2000_vals = de2000_array_from_df(X)

# Pearson R
r_model, _ = pearsonr(preds_un.ravel(), true_all.ravel())
r_de2000, _ = pearsonr(de2000_vals.ravel(), true_all.ravel())
print(f"R(model)   = {r_model:.4f}")
print(f"R(DE2000)  = {r_de2000:.4f}")

# ------------------------
# 7) Visualizations
# ------------------------
plt.figure()
plt.scatter(true_all, preds_un, s=6, alpha=0.4)
plt.xlabel("Human score (ΔE raw)")
plt.ylabel("Model prediction (ΔE raw)")
plt.title(f"Siamese Model vs Human (R={r_model:.4f})")
plt.plot([0, max(true_all.max(), preds_un.max())], [0, max(true_all.max(), preds_un.max())], 'r--', linewidth=1)
plt.savefig("scatter_siamese_pred_vs_human.png", dpi=150)
plt.close()

# error hist
err = (preds_un.ravel() - true_all.ravel())
plt.figure()
sns.histplot(err, bins=80, kde=True)
plt.title("Prediction error (pred - human)")
plt.savefig("hist_error_siamese.png", dpi=150)
plt.close()

# R comparison bar
plt.figure()
labels = ["Siamese", "ΔE2000"]
vals = [r_model, r_de2000]
sns.barplot(x=labels, y=vals)
plt.ylim(0,1)
plt.title("Pearson R comparison")
plt.savefig("r_comparison_siamese.png", dpi=150)
plt.close()

print("Saved figures: scatter_siamese_pred_vs_human.png, hist_error_siamese.png, r_comparison_siamese.png")
