"""
Master_TestingScript_vents.py

Inference / evaluation for the leak-detection models trained by
Master_ClassificationScript_vents.py.

What it does
------------
1. Rebuilds the requested architecture (dnn / cnn1d / cnn2d) from the config
   and loads the saved weights (state_dict) + the train-time standardizer.
2. Loads test features for the selected takes, optionally averaging
   consecutive frames (frame_pool), keeping every signal (clip) separate.
3. Reports accuracy, confusion matrix and ROC-AUC *per state*.
4. Applies EWMA (exponentially weighted moving average) to the model's
   per-frame leak probability, **independently for each signal in the take**,
   and reports the same metrics with and without it.
5. Plots, for one selected state, the raw vs. EWMA output of the model for
   every signal in the take.

Config: config_testing.yaml. The `model` block must match the architecture
the weights were trained with (otherwise load_state_dict will complain).
"""

import os
import math
import warnings
import numpy as np
import yaml
import torch
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             accuracy_score, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the exact model / data machinery from the training script
from Master_ClassificationScript_vents import (
    build_model, get_selected_features, pool_consecutive_frames,
    reshape_for_model, Standardizer, count_params)


# ===========================================================================
# Config
# ===========================================================================

def load_config(config_file):
    with open(config_file, "r") as stream:
        return yaml.safe_load(stream)


# ===========================================================================
# EWMA — causal, applied per signal
# ===========================================================================

def ewma(x, alpha):
    """
    Causal exponentially weighted moving average of a 1D sequence.

        s[0] = x[0]
        s[t] = alpha * x[t] + (1 - alpha) * s[t-1]

    Small alpha -> heavy smoothing / long memory (~1/alpha frames). This is
    the online form: at frame t it uses only the past, matching how a
    deployed detector accumulates evidence as frames arrive.
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.empty_like(x)
    if len(x) == 0:
        return s
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1.0 - alpha) * s[t - 1]
    return s


# ===========================================================================
# Per-signal sample extraction + model probabilities
# ===========================================================================

def clip_to_samples(clip, label, model_type, window, hop):
    """One clip -> ordered per-frame/-window/-patch samples (temporal order)."""
    samples, _ = reshape_for_model(clip[None], np.array([label]),
                                   model_type, window, hop)
    return samples


def standardize(X, mean, std):
    return X if mean is None else (X - mean) / std


def signal_probs(model, samples, mean, std, device):
    """Leak probability p(class=1) for each sample of one signal, in order."""
    X = standardize(samples.astype(np.float32), mean, std)
    X = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    with torch.no_grad():
        p = torch.softmax(model(X), dim=1)[:, 1].cpu().numpy()
    return p


# ===========================================================================
# Metrics + reporting
# ===========================================================================

def state_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "auc": auc, "cm": cm, "pred": y_pred}


def print_block(title, m):
    cm = m["cm"]
    print(f"\n  --- {title} ---")
    print(f"  accuracy {m['acc']:.4f}   ROC-AUC {m['auc']:.4f}")
    print(f"           pred_0  pred_1")
    print(f"   true_0  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"   true_1  {cm[1,0]:6d}  {cm[1,1]:6d}")


def save_state_plots(state, raw, ewma_m, y_true, p_raw, p_ewma, use_ewma,
                     out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)

    # confusion matrices (raw | ewma)
    mats = [("raw", raw["cm"])] + ([("EWMA", ewma_m["cm"])] if use_ewma else [])
    fig, axes = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 3.3),
                             squeeze=False)
    for ax, (name, cm) in zip(axes[0], mats):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["normal", "leak"])
        ax.set_yticks([0, 1], ["normal", "leak"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{state} — {name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    cm_path = os.path.join(out_dir, f"cm_{state}.png")
    fig.savefig(cm_path, dpi=150); plt.close(fig)

    # ROC overlay (raw vs ewma)
    roc_path = None
    if len(np.unique(y_true)) == 2:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        fpr, tpr, _ = roc_curve(y_true, p_raw)
        ax.plot(fpr, tpr, label=f"raw  (AUC {raw['auc']:.3f})", color="tab:blue")
        if use_ewma:
            fpr_e, tpr_e, _ = roc_curve(y_true, p_ewma)
            ax.plot(fpr_e, tpr_e, label=f"EWMA (AUC {ewma_m['auc']:.3f})",
                    color="tab:red")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(f"ROC — state '{state}'"); ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = os.path.join(out_dir, f"roc_{state}.png")
        fig.savefig(roc_path, dpi=150); plt.close(fig)
    return cm_path, roc_path


def plot_signal_outputs(sigs, state, alpha, threshold, out_dir, ncols=4):
    """
    For one state: raw vs EWMA leak-probability trace for every signal
    (clip) in the take, as a grid of subplots.
    """
    os.makedirs(out_dir, exist_ok=True)
    sigs = sorted(sigs, key=lambda s: (s["label"], s["tag"]))   # normal then leak
    n = len(sigs)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.0 * nrows),
                             squeeze=False)
    for k, sig in enumerate(sigs):
        ax = axes[k // ncols][k % ncols]
        x = np.arange(len(sig["p"]))
        ax.plot(x, sig["p"], color="tab:blue", lw=0.7, alpha=0.55, label="raw")
        ax.plot(x, sig["s"], color="tab:red", lw=1.6, label="EWMA")
        ax.axhline(threshold, ls="--", lw=0.8, color="grey")
        ax.set_ylim(-0.05, 1.05)
        is_leak = sig["label"] == 1
        ax.set_title(f"{sig['tag']}  (true={'leak' if is_leak else 'normal'})",
                     fontsize=7, color="tab:red" if is_leak else "tab:green")
        ax.tick_params(labelsize=6)
    for k in range(n, nrows * ncols):        # hide unused axes
        axes[k // ncols][k % ncols].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=8)
    fig.suptitle(f"Model leak-probability per signal — state '{state}'  "
                 f"(EWMA α={alpha}, threshold={threshold})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"signal_outputs_{state}.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


# ===========================================================================
# Main
# ===========================================================================

def main(config_file="config_testing.yaml"):
    cfg = load_config(config_file)
    d, mcfg, ecfg, ocfg = (cfg["data"], cfg["model"], cfg["ewma"], cfg["output"])
    model_type = mcfg["type"].lower().strip()
    out_dir = ocfg["results_folder"]

    device = ("cuda" if (mcfg.get("device") == "cuda" and torch.cuda.is_available())
              else "cpu")

    states_tag = "-".join(d["sel_states"])
    tag = f"{states_tag}_{d['sel_feature']}_{model_type}"
    model_path = mcfg.get("model_path") or os.path.join(out_dir, f"model_{tag}.pt")
    scaler_path = mcfg.get("scaler_path") or os.path.join(out_dir, f"scaler_{tag}.npz")

    win = mcfg.get(model_type, {}).get("window") if model_type != "dnn" else None
    hop = mcfg.get(model_type, {}).get("hop") if model_type != "dnn" else None

    alpha = ecfg.get("alpha", 0.1)
    threshold = ecfg.get("threshold", 0.5)
    use_ewma = ecfg.get("enabled", True)

    print(f"model = {model_type}   feature = {d['sel_feature']}   device = {device}")
    print(f"states = {d['sel_states']}   takes_test = {d['takes_test']}   "
          f"frame_pool = {d.get('frame_pool', 1)}")
    print(f"weights: {model_path}")

    # ---- load each signal (clip) of each state, keeping identity ---------
    #   signals[i] = dict(state, split, label, tag, samples)
    signals = []
    for state in d["sel_states"]:
        Xn, Xf = get_selected_features(
            d["feature_folder"], [state], d["sel_feature"],
            d["noises"], d["takes_test"], d["microphones"])
        Xn = pool_consecutive_frames(Xn, d.get("frame_pool", 1),
                                     d.get("frame_pool_hop", None), axis=-1)
        Xf = pool_consecutive_frames(Xf, d.get("frame_pool", 1),
                                     d.get("frame_pool_hop", None), axis=-1)
        for i in range(len(Xn)):
            samp = clip_to_samples(Xn[i], 0, model_type, win, hop)
            signals.append(dict(state=state, split="normal", label=0,
                                tag=f"{state}_iO_{i:02d}", samples=samp))
        for i in range(len(Xf)):
            samp = clip_to_samples(Xf[i], 1, model_type, win, hop)
            signals.append(dict(state=state, split="fault", label=1,
                                tag=f"{state}_niO_{i:02d}", samples=samp))

    if not signals:
        raise RuntimeError("No test signals loaded — check data config.")

    # ---- build the model and load weights --------------------------------
    sample_shape = signals[0]["samples"].shape[1:]
    model = build_model(model_type, sample_shape, mcfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    #model.load_state_dict(torch.load(model_path, map_location=device))
    
    # model_tf_mfcc_hydr_low.tflite
    model.eval()
    print(f"loaded {model_type} ({count_params(model):,} params), "
          f"sample shape {tuple(sample_shape)}")

    # ---- standardizer: prefer saved train stats --------------------------
    mean = std = None
    if mcfg.get("standardize", True):
        if os.path.isfile(scaler_path):
            z = np.load(scaler_path)
            mean, std = z["mean"], z["std"]
            print(f"scaler:  {scaler_path}")
        else:
            warnings.warn(
                f"scaler file not found at {scaler_path}; fitting on the TEST "
                f"data instead (metrics may differ slightly from training).")
            all_samples = np.concatenate([s["samples"] for s in signals], 0)
            sc = Standardizer(model_type).fit(all_samples)
            mean, std = sc.mean, sc.std

    # ---- run inference + EWMA per signal ---------------------------------
    for s in signals:
        p = signal_probs(model, s["samples"], mean, std, device)   # ordered
        s["p"] = p
        s["s"] = ewma(p, alpha) if use_ewma else p

    # ---- per-state metrics -----------------------------------------------
    summary = []
    for state in d["sel_states"]:
        st_sigs = [s for s in signals if s["state"] == state]

        y_true = np.concatenate([[s["label"]] * len(s["p"]) for s in st_sigs])
        p_raw = np.concatenate([s["p"] for s in st_sigs])
        p_ewma = np.concatenate([s["s"] for s in st_sigs])

        raw_m = state_metrics(y_true, p_raw, threshold)
        print(f"\n================  STATE: {state}  "
              f"({len(st_sigs)} signals, {len(y_true)} frames)  ================")
        print_block("raw (no EWMA)", raw_m)

        ewma_m = None
        if use_ewma:
            ewma_m = state_metrics(y_true, p_ewma, threshold)
            print_block(f"EWMA (alpha={alpha})", ewma_m)
            print(f"\n  delta:  accuracy {ewma_m['acc']-raw_m['acc']:+.4f}   "
                  f"AUC {ewma_m['auc']-raw_m['auc']:+.4f}")

        if ocfg.get("save_plots", True):
            cm_p, roc_p = save_state_plots(state, raw_m, ewma_m, y_true,
                                           p_raw, p_ewma, use_ewma, out_dir,
                                           threshold)
            print(f"  saved: {cm_p}" + (f" , {roc_p}" if roc_p else ""))

        summary.append((state, raw_m, ewma_m))

    # ---- per-signal output plot for the chosen state ---------------------
    plot_state = ocfg.get("plot_state")
    if plot_state:
        st_sigs = [s for s in signals if s["state"] == plot_state]
        if st_sigs:
            path = plot_signal_outputs(st_sigs, plot_state, alpha, threshold,
                                       out_dir, ocfg.get("plot_ncols", 4))
            print(f"\nper-signal output plot -> {path}")
        else:
            print(f"\nplot_state '{plot_state}' not among sel_states; skipped.")

    # ---- compact recap ----------------------------------------------------
    print("\n================  SUMMARY  ================")
    hdr = f"{'state':12s}  {'raw_acc':>8s} {'raw_auc':>8s}"
    if use_ewma:
        hdr += f"   {'ewma_acc':>8s} {'ewma_auc':>8s}"
    print(hdr)
    for state, raw_m, ewma_m in summary:
        line = f"{state:12s}  {raw_m['acc']:8.4f} {raw_m['auc']:8.4f}"
        if use_ewma:
            line += f"   {ewma_m['acc']:8.4f} {ewma_m['auc']:8.4f}"
        print(line)


if __name__ == "__main__":
    main()



# %%
"""

================  SUMMARY  ================
================  SUMMARY  ================
state          raw_acc  raw_auc   ewma_acc ewma_auc
tubeleak        0.6266   0.7340     0.6959   0.8572
ventleak        0.8582   0.9537     0.8701   0.9510
ventlow         0.9123   0.9574     0.9383   0.9600
"""
