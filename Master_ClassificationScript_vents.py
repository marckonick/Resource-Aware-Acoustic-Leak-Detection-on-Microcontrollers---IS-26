"""
Master_ClassificationScript_vents.py

Train and test lightweight DNN / 1D-CNN / 2D-CNN leak-vs-no-leak classifiers
on the features produced by FeatureExtractrionFunctions.py.

Everything is driven by config_classification.yaml:
  * which features to load (state / feature / noises / mics)
  * which *takes* are used for training vs. testing  (take-based split ->
    no clip is ever shared between train and test)
  * model type and its layer/neuron layout
  * optimisation hyper-parameters

How each saved feature array becomes training samples
-----------------------------------------------------
Saved shapes (from the extraction stage):
    spectral_summary / bandpower / mfcc :  (N_clips, D, T)
    logmel                              :  (N_clips, n_mels, T)
    logmel_patch                        :  (N_clips, num_patches, n_mels, patch)

  dnn    -> per-frame samples          (N*T, D)          [one frame = one sample]
  cnn1d  -> time windows               (N*num_win, D, W) [D channels over time]
  cnn2d  -> patches / spectro windows  (N*num_pat, 1, H, W)

The output layer always has 2 neurons (CrossEntropyLoss on logits).
"""


import os
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             accuracy_score, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Config
# ===========================================================================

def load_config(config_file):
    with open(config_file, "r") as stream:
        return yaml.safe_load(stream)


# ===========================================================================
# Loading saved features
# ===========================================================================

def get_selected_features(feature_folder, sel_states, sel_feature,
                          noises, takes, microphones):
    """
    Load and concatenate normal / fault feature arrays across the sweep.

    `sel_states` is a list (e.g. ["tubeleak", "ventleak"]); every state's
    normal clips are pooled into the normal class and every state's fault
    clips into the fault class. Feature arrays share (D, T) / (H, W) shape,
    so they concatenate cleanly along the clip axis.
    """
    if isinstance(sel_states, str):          # tolerate a single string
        sel_states = [sel_states]
    X_n, X_f = [], []
    for sel_state in sel_states:
        for noise in noises:
            for take in takes:
                for microphone in microphones:
                    base = os.path.join(
                        feature_folder,
                        f"{sel_state}_{{split}}_{sel_feature}_{noise}_t{take}_m{microphone}.npy")
                    fn = base.format(split="normal")
                    ff = base.format(split="fault")
                    if os.path.isfile(fn):
                        X_n.append(np.load(fn))
                    if os.path.isfile(ff):
                        X_f.append(np.load(ff))
    if not X_n or not X_f:
        raise FileNotFoundError(
            f"No feature files found for states={sel_states}, takes={takes}. "
            f"Checked folder '{feature_folder}' with feature '{sel_feature}'.")
    return np.concatenate(X_n, 0), np.concatenate(X_f, 0)


def pool_consecutive_frames(X, pool, hop=None, axis=-1):
    """
    Average `pool` consecutive frames along the time axis with stride `hop`.

    Reduces temporal resolution right after loading (a cheap smoothing /
    downsampling step). pool None or <= 1 -> returned unchanged. A trailing
    partial window is dropped. For (N, D, T) features this pools T; for 4D
    (N, P, H, W) patch features it pools the within-patch frame axis W.
    """
    if pool is None or pool <= 1:
        return X
    if hop is None:
        hop = pool
    
    axis = 1 if X.ndim == 4 else -1
    
    Xm = np.moveaxis(X, axis, -1)                 # time -> last axis
    T = Xm.shape[-1]
    starts = range(0, T - pool + 1, hop)
    pooled = np.stack([Xm[..., s:s + pool].mean(axis=-1) for s in starts], axis=-1)
    return np.moveaxis(pooled, -1, axis)


def build_xy(feature_folder, d, takes):
    """Clip-level (X, y): label 0 = normal (no leak), 1 = fault (leak).

    Pools across all `sel_states`, then optionally averages consecutive
    frames (frame_pool / frame_pool_hop) before the samples are reshaped.
    """
    Xn, Xf = get_selected_features(feature_folder, d["sel_states"], d["sel_feature"],
        d["noises"], takes, d["microphones"])
    X = np.concatenate([Xn, Xf], 0)
    y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xf))]).astype(np.int64)

    pool_axis = 1 if X.ndim == 4 else -1
    
    X = pool_consecutive_frames(
        X, d.get("frame_pool", 1), d.get("frame_pool_hop", None), axis=pool_axis)
    return X, y


# ===========================================================================
# Turning clip arrays into model-ready samples
# ===========================================================================

def _window_along_time(X, y, window, hop):
    """(N, C, T) -> (N*num_win, C, window). Trailing partial window dropped."""
    N, C, T = X.shape
    if window is None:
        return X, y
    if hop is None:
        hop = window
    starts = range(0, T - window + 1, hop)
    chunks = [X[:, :, s:s + window] for s in starts]
    labels = [y for _ in starts]
    return np.concatenate(chunks, 0), np.concatenate(labels, 0)


def reshape_for_model(X, y, model_type, window=None, hop=None):
    """
    Expand clip-level arrays into per-sample tensors for the chosen model.
    Handles 2D (already summarised), 3D (N, C, T) and 4D (N, P, H, W) inputs.
    """
    X = np.asarray(X, dtype=np.float32)

    if model_type == "dnn":
        if X.ndim == 3:                       # (N, D, T) -> (N*T, D)
            N, D, T = X.shape
            Xr = np.transpose(X, (0, 2, 1)).reshape(N * T, D)
            yr = np.repeat(y, T)
        elif X.ndim == 4:                     # (N, P, H, W) -> (N*P, H*W)
            N, P, H, W = X.shape
            Xr = X.reshape(N * P, H * W)
            yr = np.repeat(y, P)
        else:                                 # (N, D) already summarised
            Xr, yr = X, y
        return Xr, yr

    if model_type == "cnn1d":
        assert X.ndim == 3, "cnn1d expects (N, D, T) features"
        Xr, yr = _window_along_time(X, y, window, hop)      # (num, D, W)
        return Xr, yr

    if model_type == "cnn2d":
        if X.ndim == 4:                       # (N, P, H, W) -> (N*P, 1, H, W)
            N, P, H, W = X.shape
            Xr = X.reshape(N * P, 1, H, W)
            yr = np.repeat(y, P)
        elif X.ndim == 3:                     # (N, H, T) -> window -> (num,1,H,W)
            Xw, yr = _window_along_time(X, y, window, hop)
            Xr = Xw[:, None, :, :]
        else:
            raise ValueError("cnn2d expects 3D spectrogram or 4D patch features")
        return Xr, yr

    raise ValueError(f"Unknown model type: {model_type!r}")


class Standardizer:
    """z-score using train statistics; keeps the channel structure intact."""
    def __init__(self, model_type):
        self.model_type = model_type
        self.mean = None
        self.std = None

    def fit(self, X):
        if self.model_type == "dnn":            # (num, D)     -> per feature
            axes = (0,)
        elif self.model_type == "cnn1d":        # (num, C, T)  -> per channel
            axes = (0, 2)
        else:                                   # (num,1,H,W)  -> per mel band
            axes = (0, 1, 3)
        self.mean = X.mean(axis=axes, keepdims=True)
        self.std = X.std(axis=axes, keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


# ===========================================================================
# Models
# ===========================================================================

class DNN(nn.Module):
    """Stack of Linear + ReLU (+Dropout), then a 2-way classification layer."""
    def __init__(self, in_dim, hidden_layers, n_classes=2, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden_layers:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_classes)]     # output layer: 2 neurons
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Conv1d stack -> flatten -> optional Linear layers -> 2-way output."""
    def __init__(self, in_ch, in_len, conv_channels, kernel_size, pool_size,
                 fc_layers, n_classes=2, dropout=0.0):
        super().__init__()
        convs, c = [], in_ch
        for oc in conv_channels:
            convs += [nn.Conv1d(c, oc, kernel_size, padding=kernel_size // 2),
                      nn.ReLU(), nn.MaxPool1d(pool_size)]
            c = oc
        self.conv = nn.Sequential(*convs)
        with torch.no_grad():                   # infer flattened size
            flat = self.conv(torch.zeros(1, in_ch, in_len)).numel()
        self.head = _make_head(flat, fc_layers, n_classes, dropout)

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class CNN2D(nn.Module):
    """Frequency-preserving tiny 2D CNN for log-mel inputs.

    Input shape
    -----------
    ``(batch, 1, n_mels, time_frames)``

    The previous CNN pooled and globally averaged across both frequency and
    time. That makes the representation partly invariant to the absolute mel
    band in which a pattern occurs. For leak detection, absolute frequency
    position can be important.

    This implementation therefore:
      1. applies small 2D convolutions;
      2. max-pools only along time, never along frequency;
      3. averages only along the remaining time axis;
      4. flattens ``channels x mel_bands`` for the small dense head.

    With a 40 x 24 input, ``conv_channels=[4, 8]`` and ``fc_layers=[4]``,
    the network has 1,630 trainable parameters for a two-class output.
    """

    def __init__(
        self,
        in_ch,
        in_hw,
        conv_channels,
        kernel_size,
        pool_size,
        fc_layers,
        n_classes=2,
        dropout=0.0,
    ):
        super().__init__()

        if not conv_channels:
            raise ValueError("cnn2d.conv_channels must contain at least one layer")

        if isinstance(pool_size, int):
            time_pool = pool_size
        elif isinstance(pool_size, (tuple, list)) and len(pool_size) == 2:
            if int(pool_size[0]) != 1:
                raise ValueError(
                    "Frequency-preserving CNN requires pool_size[0] == 1; "
                    "pooling is allowed only along time."
                )
            time_pool = int(pool_size[1])
        else:
            raise ValueError("cnn2d.pool_size must be an int or a two-element sequence")

        if time_pool < 1:
            raise ValueError("cnn2d.pool_size must be >= 1")

        convs = []
        c = in_ch
        for oc in conv_channels:
            convs += [
                nn.Conv2d(
                    in_channels=c,
                    out_channels=oc,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
                # Preserve all mel bands; reduce only temporal resolution.
                nn.MaxPool2d(
                    kernel_size=(1, time_pool),
                    stride=(1, time_pool),
                ),
            ]
            c = oc

        self.conv = nn.Sequential(*convs)

        # Infer the dense-head input size. After temporal mean pooling, the
        # remaining representation is (channels, frequency_bins).
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, *in_hw)
            z = self.conv(dummy)
            if z.shape[-1] < 1:
                raise ValueError(
                    "Temporal dimension became empty. Reduce the number of "
                    "CNN layers or the temporal pool size."
                )
            z = z.mean(dim=-1)             # average time only -> (1, C, H)
            flat = z.flatten(1).shape[1]   # C * H; frequency is retained

        self.head = _make_head(
            flat=flat,
            fc_layers=fc_layers,
            n_classes=n_classes,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.conv(x)                   # (B, C, n_mels, reduced_time)
        x = x.mean(dim=-1)                 # temporal average only: (B, C, n_mels)
        x = torch.flatten(x, start_dim=1)  # (B, C * n_mels)
        return self.head(x)


def _make_head(flat, fc_layers, n_classes, dropout):
    layers, d = [], flat
    for h in fc_layers:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, n_classes)]         # output layer: 2 neurons
    return nn.Sequential(*layers)
    



def build_model(model_type, sample_shape, mcfg):
    """Instantiate the model from the prepared-sample shape (minus batch)."""
    n_classes = mcfg.get("n_classes", 2)
    if model_type == "dnn":
        c = mcfg["dnn"]
        return DNN(sample_shape[0], c["hidden_layers"], n_classes,
                   c.get("dropout", 0.0))
    if model_type == "cnn1d":
        c = mcfg["cnn1d"]
        return CNN1D(sample_shape[0], sample_shape[1], c["conv_channels"],
                     c["kernel_size"], c["pool_size"], c["fc_layers"],
                     n_classes, c.get("dropout", 0.0))
    if model_type == "cnn2d":
        c = mcfg["cnn2d"]
        return CNN2D(
            in_ch=sample_shape[0],
            in_hw=(sample_shape[1], sample_shape[2]),
            conv_channels=c["conv_channels"],
            kernel_size=c["kernel_size"],
            pool_size=c["pool_size"],
            fc_layers=c["fc_layers"],
            n_classes=n_classes,
            dropout=c.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown model type: {model_type!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Training / evaluation
# ===========================================================================

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                       torch.from_numpy(np.ascontiguousarray(y)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, tr_loader, val_loader, tcfg, device, class_weights=None):
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                           weight_decay=tcfg.get("weight_decay", 0.0))
    crit = nn.CrossEntropyLoss(
        weight=None if class_weights is None
        else torch.tensor(class_weights, dtype=torch.float32, device=device))

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += len(yb)

        msg = f"epoch {epoch:3d}/{tcfg['epochs']}  " \
              f"train_loss {tr_loss / tr_total:.4f}  acc {tr_correct / tr_total:.3f}"
        if val_loader is not None:
            vl, va = _eval_loss_acc(model, val_loader, crit, device)
            msg += f"   |   val_loss {vl:.4f}  acc {va:.3f}"
        print(msg)
    return model


def _eval_loss_acc(model, loader, crit, device):
    model.eval()
    loss = correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss += crit(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss / total, correct / total


def predict(model, loader, device):
    """Return (y_true, y_pred, prob_of_class_1)."""
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds.append(logits.argmax(1).cpu().numpy())
            probs.append(p)
            ys.append(yb.numpy())
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(probs))


# ===========================================================================
# Reporting
# ===========================================================================

def report(y_true, y_pred, y_prob, out_dir, tag, save_plots=True):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")      # only one class present in the test set

    print("\n================  TEST RESULTS  ================")
    print(f"samples: {len(y_true)}   accuracy: {acc:.4f}   ROC-AUC: {auc:.4f}")
    print("confusion matrix  [rows = true, cols = pred]  (0=normal, 1=leak)")
    print(f"            pred_0   pred_1")
    print(f"  true_0   {cm[0, 0]:7d}  {cm[0, 1]:7d}")
    print(f"  true_1   {cm[1, 0]:7d}  {cm[1, 1]:7d}")
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=["normal", "leak"],
                                       digits=4, zero_division=0))

    if save_plots:
        # confusion matrix
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["normal", "leak"])
        ax.set_yticks([0, 1], ["normal", "leak"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix  ({tag})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        cm_path = os.path.join(out_dir, f"confusion_matrix_{tag}.png")
        fig.savefig(cm_path, dpi=150); plt.close(fig)

        # ROC curve
        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
            ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
            ax.set_title(f"ROC  ({tag})"); ax.legend(loc="lower right")
            fig.tight_layout()
            roc_path = os.path.join(out_dir, f"roc_curve_{tag}.png")
            fig.savefig(roc_path, dpi=150); plt.close(fig)
            print(f"saved plots -> {cm_path} , {roc_path}")
        else:
            print(f"saved plot  -> {cm_path}  (ROC skipped: one class only)")

    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm}


# ===========================================================================
# Main
# ===========================================================================

def main(config_file="config_classification.yaml"):
    cfg = load_config(config_file)
    d, mcfg, tcfg, ocfg = (cfg["data"], cfg["model"],
                           cfg["training"], cfg["output"])
    model_type = mcfg["type"].lower().strip()

    seed = tcfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = ("cuda" if (tcfg.get("device") == "cuda" and torch.cuda.is_available())
              else "cpu")
    print(f"model = {model_type}   feature = {d['sel_feature']}   device = {device}")
    print(f"states = {d['sel_states']}   frame_pool = {d.get('frame_pool', 1)}"
          f" (hop {d.get('frame_pool_hop', None)})")
    print(f"takes_train = {d['takes_train']}   takes_test = {d['takes_test']}")

    # windowing params (only used by the conv models)
    win = mcfg.get(model_type, {}).get("window") if model_type != "dnn" else None
    hop = mcfg.get(model_type, {}).get("hop") if model_type != "dnn" else None

    # ---- load clip-level data, split by take -----------------------------
    Xtr_clip, ytr_clip = build_xy(d["feature_folder"], d, d["takes_train"])
    Xte_clip, yte_clip = build_xy(d["feature_folder"], d, d["takes_test"])

    # clip-level train/val split so overlapping windows never leak across it
    val_split = tcfg.get("val_split", 0.0)
    if val_split and val_split > 0:
        tr_idx, val_idx = train_test_split(
            np.arange(len(ytr_clip)), test_size=val_split,
            random_state=seed, stratify=ytr_clip)
    else:
        tr_idx, val_idx = np.arange(len(ytr_clip)), np.array([], dtype=int)

    # ---- expand clips into model-ready samples ---------------------------
    Xtr, ytr = reshape_for_model(Xtr_clip[tr_idx], ytr_clip[tr_idx],
                                 model_type, win, hop)
    Xte, yte = reshape_for_model(Xte_clip, yte_clip, model_type, win, hop)
    if len(val_idx):
        Xval, yval = reshape_for_model(Xtr_clip[val_idx], ytr_clip[val_idx],
                                       model_type, win, hop)

    # ---- standardise using train statistics ------------------------------
    scaler = None
    if tcfg.get("standardize", True):
        scaler = Standardizer(model_type).fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
        if len(val_idx):
            Xval = scaler.transform(Xval)

    print(f"train samples {Xtr.shape}   test samples {Xte.shape}")

    # ---- loaders ---------------------------------------------------------
    bs = tcfg["batch_size"]
    tr_loader = make_loader(Xtr, ytr, bs, shuffle=True)
    te_loader = make_loader(Xte, yte, bs, shuffle=False)
    val_loader = make_loader(Xval, yval, bs, shuffle=False) if len(val_idx) else None

    # ---- optional class weights ------------------------------------------
    class_weights = None
    if tcfg.get("class_weight", False):
        counts = np.bincount(ytr, minlength=2).astype(np.float32)
        class_weights = (counts.sum() / (2.0 * np.maximum(counts, 1))).tolist()
        print(f"class weights: {class_weights}")

    # ---- build, train, test ----------------------------------------------
    model = build_model(model_type, Xtr.shape[1:], mcfg).to(device)
    print(f"{model_type} parameters: {count_params(model):,}")
    print(model)

    model = train_model(model, tr_loader, val_loader, tcfg, device, class_weights)

    y_true, y_pred, y_prob = predict(model, te_loader, device)
    states_tag = "-".join(d["sel_states"])
    tag = f"{states_tag}_{d['sel_feature']}_{model_type}"
    report(y_true, y_pred, y_prob, ocfg["results_folder"], tag,
           ocfg.get("save_plots", True))

    if ocfg.get("save_model", True):
        os.makedirs(ocfg["results_folder"], exist_ok=True)
        mpath = os.path.join(ocfg["results_folder"], f"model_{tag}.pt")
        torch.save(model.state_dict(), mpath)
        print(f"saved model -> {mpath}")
        if scaler is not None:      # save train-time stats for the testing script
            spath = os.path.join(ocfg["results_folder"], f"scaler_{tag}.npz")
            np.savez(spath, mean=scaler.mean, std=scaler.std)
            print(f"saved scaler -> {spath}")


if __name__ == "__main__":
    main()
    
    
    
    
    
    