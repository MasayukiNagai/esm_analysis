# esm_helpers.py
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ---------- palettes ----------
def make_palettes(n=256):
    cmap = plt.colormaps["bwr_r"]
    bwr_r = [to_hex(cmap(i)) for i in np.linspace(0, 1, n)]
    cmap = plt.colormaps["gray_r"]
    gray = [to_hex(cmap(i)) for i in np.linspace(0, 1, n)]
    return bwr_r, gray


# ---------- dataframe helpers ----------
def pssm_to_dataframe(pssm, esm_alphabet):
    sequence_length = pssm.shape[0]
    idx = [str(i) for i in range(1, sequence_length + 1)]
    df = pd.DataFrame(pssm, index=idx, columns=list(esm_alphabet))
    df = df.stack().reset_index()
    df.columns = ["Position", "Amino Acid", "Probability"]
    return df


def contact_to_dataframe(con):
    sequence_length = con.shape[0]
    idx = [str(i) for i in range(1, sequence_length + 1)]
    df = pd.DataFrame(con, index=idx, columns=idx)
    df = df.stack().reset_index()
    df.columns = ["i", "j", "value"]
    return df


def pair_to_dataframe(pair, esm_alphabet):
    df = pd.DataFrame(pair, index=list(esm_alphabet), columns=list(esm_alphabet))
    df = df.stack().reset_index()
    df.columns = ["aa_i", "aa_j", "value"]
    return df


# ---------- model utilities ----------
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_name, device=None):
    if device is None:
        device = get_device()
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    model = model.to(device).eval()
    return model, alphabet, device


def get_logits(model, alphabet, device, seq, p=1, return_jac=False):
    x, ln = alphabet.get_batch_converter()([(None, seq)])[-1], len(seq)
    if p is None:
        p = ln

    with torch.no_grad():
        f = lambda x: model(x)["logits"][:, 1:(ln + 1), 4:24].detach().cpu().numpy()
        logits = np.zeros((ln, 20), dtype=np.float32)

        if return_jac:
            jac = np.zeros((ln, 1, ln, 20), dtype=np.float32)
            fx = f(x.to(device))[0]

        with tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
            for n in range(0, ln, p):
                m = min(n + p, ln)
                x_h = torch.tile(torch.clone(x), [m - n, 1])
                for i in range(m - n):
                    x_h[i, n + i + 1] = alphabet.mask_idx
                fx_h = f(x_h.to(device))
                for i in range(m - n):
                    logits[n + i] = fx_h[i, n + i]
                    if return_jac:
                        jac[n + i] = fx_h[i, None] - fx
                pbar.update(m - n)

        return jac if return_jac else logits


def get_categorical_jacobian(model, alphabet, device, seq, layer=None, fast=False):
    x, ln = alphabet.get_batch_converter()([("seq", seq)])[-1], len(seq)

    with torch.no_grad():
        if layer is None:
            f = lambda x: model(x)["logits"][..., 1:(ln + 1), 4:24].detach().cpu().numpy()
        else:
            f = lambda x: model(
                x, repr_layers=[layer]
            )["representations"][layer][..., 1:(ln + 1), :].detach().cpu().numpy()

        fx = f(x.to(device))[0]
        fx_h = np.zeros((ln, 1 if fast else 20, ln, fx.shape[-1]), dtype=np.float32)
        x_ = x.to(device) if fast else torch.tile(x, [20, 1]).to(device)

        with tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
            for n in range(ln):
                x_h = torch.clone(x_)
                x_h[:, n + 1] = (
                    alphabet.mask_idx if fast else torch.arange(4, 24, device=x_h.device)
                )
                fx_h[n] = f(x_h)
                pbar.update(1)

    return fx_h - fx


def jac_to_con(
    jac,
    esm_alphabet,
    ALPHABET,
    center=True,
    diag="remove",
    apc=True,
    symm=True,
):
    X = jac.copy()
    _, Ax, _, Ay = X.shape

    ALPHABET_map = [esm_alphabet.index(a) for a in ALPHABET]

    if Ax == 20:
        X = X[:, ALPHABET_map, :, :]
    if Ay == 20:
        X = X[:, :, :, ALPHABET_map]
        if symm and Ax == 20:
            X = (X + X.transpose(2, 3, 0, 1)) / 2

    if center:
        for i in range(4):
            if X.shape[i] > 1:
                X -= X.mean(i, keepdims=True)

    contacts = np.sqrt(np.square(X).sum((1, 3)))

    if symm and (Ax != 20 or Ay != 20):
        contacts = (contacts + contacts.T) / 2

    if diag == "remove":
        np.fill_diagonal(contacts, 0)

    if diag == "normalize":
        d = np.diag(contacts)
        contacts = contacts / np.sqrt(d[:, None] * d[None, :])

    if apc:
        ap = contacts.sum(0, keepdims=True) * contacts.sum(1, keepdims=True) / contacts.sum()
        contacts = contacts - ap

    if diag == "remove":
        np.fill_diagonal(contacts, 0)

    return {"jac": X, "contacts": contacts}
