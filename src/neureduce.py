"""Torchtext-free loader for NeuReduce's PyTorch Transformer (PyTorch/best-transformer.pt).

Reproduces what PyTorch/transformer.ipynb does, without the legacy torchtext API:
  * character-level tokens with <sos>/<eos>, vocab built from the training split
    in legacy-torchtext order (<unk>, <pad>, <sos>, <eos>, then chars by frequency desc, ties alphabetical)
  * the 80/20 split of PyTorch/dataset.csv under random.seed(1234), i.e. the same held-out
    examples the notebook calls `valid_data` (the checkpoint never trained on these)
The model classes are copied from the notebook unchanged (only shape comments removed), so the
state_dict loads with strict=True.
"""
import ast
import csv
import itertools
import operator
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

SEED = 1234
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK, PAD, SOS, EOS = 0, 1, 2, 3
MAX_POSITIONS = 100  # size of the learned positional embedding -> at most 98 chars + <sos>/<eos>


# --------------------------------------------------------------------------- data

def read_pairs(csv_path):
    """dataset.csv has no header: obfuscated_expr,simplified_expr"""
    with open(csv_path, newline="") as f:
        return [(src, trg) for src, trg in csv.reader(f)]


def notebook_split(n, ratio=0.8, seed=SEED):
    """Indices of torchtext's Dataset.split(split_ratio=0.8) right after random.seed(1234).
    Spelled out (= random.sample(range(n), n) on CPython 3.7-3.12) so it can't drift with Python versions."""
    rng = random.Random(seed)

    def randbelow(m):
        k = m.bit_length()
        r = rng.getrandbits(k)
        while r >= m:
            r = rng.getrandbits(k)
        return r

    pool, perm = list(range(n)), [0] * n
    for i in range(n):
        j = randbelow(n - i)
        perm[i] = pool[j]
        pool[j] = pool[n - i - 1]
    n_train = int(round(ratio * n))
    return perm[:n_train], perm[n_train:]


class Vocab:
    def __init__(self, sequences):
        counts = Counter(ch for s in sequences for ch in s.lower())
        chars = sorted(counts.items(), key=lambda t: t[0])
        chars.sort(key=lambda t: t[1], reverse=True)  # stable: frequency desc, ties alphabetical
        self.itos = SPECIALS + [c for c, _ in chars]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.freqs = counts

    def __len__(self):
        return len(self.itos)

    def encode(self, expr):
        expr = expr.replace(" ", "").lower()
        if len(expr) + 2 > MAX_POSITIONS:
            raise ValueError(f"expression has {len(expr)} chars; the model only has {MAX_POSITIONS} positions")
        return [SOS] + [self.stoi.get(ch, UNK) for ch in expr] + [EOS]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i not in (PAD, SOS, EOS))


# --------------------------------------------------------------------------- model (verbatim from transformer.ipynb)

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)  # [batch, heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention  # attention = cross-attention of the LAST decoder layer only


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention


# --------------------------------------------------------------------------- loading

def load(neureduce_dir="NeuReduce-original", checkpoint="best-transformer.pt", device=None):
    """Returns (model, src_vocab, trg_vocab, pairs, train_idx, test_idx). Model is in eval mode."""
    d = Path(neureduce_dir) / "PyTorch"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = read_pairs(d / "dataset.csv")
    train_idx, test_idx = notebook_split(len(pairs))
    src_vocab = Vocab(pairs[i][0] for i in train_idx)
    trg_vocab = Vocab(pairs[i][1] for i in train_idx)

    # hyper-parameters from the notebook (HID_DIM=256, 3+3 layers, 8 heads, PF_DIM=512, dropout 0.1)
    enc = Encoder(len(src_vocab), 256, 3, 8, 512, 0.1, device)
    dec = Decoder(len(trg_vocab), 256, 3, 8, 512, 0.1, device)
    model = Seq2Seq(enc, dec, PAD, PAD, device).to(device)
    state = torch.load(d / checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()  # disables dropout -- forgetting this gives noisy, worse predictions
    return model, src_vocab, trg_vocab, pairs, train_idx, test_idx


# --------------------------------------------------------------------------- inference

@torch.inference_mode()
def predict(model, exprs, src_vocab, trg_vocab, batch_size=256, max_len=50):
    """Batched greedy decoding (same decisions as the notebook's translate(), just faster).
    Returns one dict per input: pred, logprob (sum over generated tokens incl. <eos>),
    min_tok_prob (least confident step), finished (emitted <eos> before max_len)."""
    device = model.device
    out = []
    for b in range(0, len(exprs), batch_size):
        chunk = [src_vocab.encode(e) for e in exprs[b:b + batch_size]]
        width = max(map(len, chunk))
        src = torch.tensor([s + [PAD] * (width - len(s)) for s in chunk], device=device)
        src_mask = model.make_src_mask(src)
        enc_src = model.encoder(src, src_mask)

        n = len(chunk)
        trg = torch.full((n, 1), SOS, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        logprob = torch.zeros(n, device=device)
        min_prob = torch.ones(n, device=device)
        for _ in range(max_len):
            logits, _ = model.decoder(trg, enc_src, model.make_trg_mask(trg), src_mask)
            logp = torch.log_softmax(logits[:, -1], dim=-1)
            step_logp, nxt = logp.max(dim=-1)
            logprob += torch.where(done, 0.0, step_logp)
            min_prob = torch.where(done, min_prob, torch.minimum(min_prob, step_logp.exp()))
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            trg = torch.cat([trg, nxt[:, None]], dim=1)
            done |= nxt == EOS
            if done.all():
                break
        for i in range(n):
            ids = trg[i, 1:].tolist()
            ids = ids[:ids.index(EOS)] if EOS in ids else ids
            out.append({"pred": trg_vocab.decode(ids), "logprob": logprob[i].item(),
                        "min_tok_prob": min_prob[i].item(), "finished": bool(done[i])})
    return out


# --------------------------------------------------------------------------- correctness (paper: exact OR equivalent)
# Plain Z3 is slow on MBA (that's the point of MBA), so: refute with random 64-bit tests, prove linear MBA via the
# {0,-1}-corner (truth-table signature) theorem, and only fall back to Z3 for non-linear outputs.

_ALPHABET = re.compile(r"[a-z0-9&|^~+\-*()]+")
_PY_ONLY = re.compile(r"[0-9][a-z]|[a-z][a-z0-9]")  # Python-only literals (0x1f, 1e5) / multi-letter names
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}
_MASK = (1 << 64) - 1


def parse(expr):
    """MBA string -> Python AST (C and Python agree on precedence: unary > * > +- > & > ^ > |)."""
    if not _ALPHABET.fullmatch(expr) or _PY_ONLY.search(expr):
        raise SyntaxError(expr)
    tree = ast.parse(expr, mode="eval").body
    for n in ast.walk(tree):
        ok = isinstance(n, (ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, *_BIN, *_UN)) or \
            (isinstance(n, ast.Constant) and type(n.value) is int)
        if not ok:
            raise SyntaxError(expr)
    return tree


def evaluate(node, env):
    """Works for Python ints and z3 bit-vectors alike."""
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](evaluate(node.left, env), evaluate(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UN[type(node.op)](evaluate(node.operand, env))
    if isinstance(node, ast.Name):
        return env[node.id]
    return node.value


def _bitwise(n):
    if isinstance(n, ast.Name):
        return True
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Invert):
        return _bitwise(n.operand)
    return isinstance(n, ast.BinOp) and isinstance(n.op, (ast.BitAnd, ast.BitOr, ast.BitXor)) \
        and _bitwise(n.left) and _bitwise(n.right)


def _const(n):
    return isinstance(n, ast.Constant) or \
        (isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)) and _const(n.operand))


def is_linear(n):
    """sum of constant * (pure bitwise term over variables) + constant"""
    if _bitwise(n) or _const(n):
        return True
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        return is_linear(n.operand)
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
        return is_linear(n.left) and is_linear(n.right)
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
        return (_const(n.left) and is_linear(n.right)) or (_const(n.right) and is_linear(n.left))
    return False


def verdict(pred, trg, n_random=64, z3_timeout_ms=2000):
    """'exact' | 'equivalent' | 'wrong' | 'invalid' (unparseable output) | 'unknown' (non-linear, Z3 timed out)"""
    if pred == trg:
        return "exact"
    try:
        p = parse(pred)
    except Exception:
        return "invalid"
    t = parse(trg)
    names = sorted(set(re.findall(r"[a-z]", pred + trg)))
    corners = [dict(zip(names, bits)) for bits in itertools.product((0, -1), repeat=len(names))]
    rng = random.Random(0)
    rand = [{v: rng.getrandbits(64) for v in names} for _ in range(n_random)]
    for env in corners + rand:  # any mismatch is a definite counterexample
        if (evaluate(p, env) - evaluate(t, env)) & _MASK:
            return "wrong"
    if is_linear(p) and is_linear(t):  # equal on all {0,-1} corners => equal for every bit-width
        return "equivalent"
    import z3
    env = {v: z3.BitVec(v, 64) for v in names}
    a, b = (evaluate(x, env) for x in (p, t))
    a, b = (x if isinstance(x, z3.BitVecRef) else z3.BitVecVal(x, 64) for x in (a, b))
    s = z3.Solver()
    s.set("timeout", z3_timeout_ms)
    s.add(a != b)
    r = s.check()
    if r == z3.unsat:
        return "equivalent"
    return "wrong" if r == z3.sat else "unknown"


# --------------------------------------------------------------------------- structural attributes for the analysis CSV

_OPS = {"and": "&", "or": "|", "xor": "^", "not": "~", "add": "+", "sub": "-", "mul": "*"}


def attributes(expr):
    """Cheap string-level features (work on unparseable outputs too). 'sub' counts every '-' incl. unary minus."""
    depth = cur = terms = 0
    for i, ch in enumerate(expr):
        cur += ch == "("
        cur -= ch == ")"
        depth = max(depth, cur)
        if ch in "+-" and cur == 0 and i > 0 and expr[i - 1] not in "+-*&|^~(":
            terms += 1  # binary +/- at top level separates terms
    coefs = [int(c) for c in re.findall(r"\d+", expr)]
    return {"len": len(expr), "n_vars": len(set(re.findall(r"[a-z]", expr))),
            "n_terms": terms + 1 if expr else 0, "max_depth": depth, "max_const": max(coefs, default=0),
            **{f"n_{name}": expr.count(sym) for name, sym in _OPS.items()}}
