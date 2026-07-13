"""
H-JEPA-LM Benchmark: Hierarchical JEPA + Action Conditioning
=============================================================
Compare: Base JEPA-LM vs H-JEPA-LM vs LLM-JEPA vs BERT-MLM vs GPT-NTP
"""
import os, json, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
RESULTS_DIR = "./benchmark_hjepa_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class Block(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim), nn.Dropout(dropout))
    def forward(self, x, causal=False):
        B, N, _ = x.shape; r = x; xn = self.n1(x)
        if causal:
            m = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
            o, _ = self.attn(xn, xn, xn, attn_mask=m)
        else:
            o, _ = self.attn(xn, xn, xn)
        return r + o + self.ff(self.n2(r + o))


# ============================================================
# GPT-NTP
# ============================================================
class GPTModel(nn.Module):
    def __init__(self, V, D, L, H, F, M):
        super().__init__(); self.dim = D
        self.te = nn.Embedding(V, D); self.pe = nn.Embedding(M, D)
        self.ly = nn.ModuleList([Block(D, H, F) for _ in range(L)]); self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V); self.head.weight = self.te.weight
    def encode(self, ids):
        B, N = ids.shape; p = torch.arange(N, device=ids.device).unsqueeze(0).expand(B,-1)
        h = self.te(ids) + self.pe(p)
        for l in self.ly: h = l(h, causal=True)
        return self.norm(h)
    def get_embedding(self, h): return h[:, -1, :]
    def forward(self, ids): return self.head(self.encode(ids))
    def compute_loss(self, ids):
        lo = self.forward(ids); t = ids[:,1:].contiguous(); lo = lo[:,:-1,:].contiguous()
        return F.cross_entropy(lo.reshape(-1, lo.size(-1)), t.reshape(-1))
    def update_ema(self, m): pass


# ============================================================
# BERT-MLM
# ============================================================
class BERTModel(nn.Module):
    def __init__(self, V, D, L, H, F, M):
        super().__init__(); self.dim = D; self.V = V
        self.te = nn.Embedding(V, D); self.pe = nn.Embedding(M, D)
        self.ly = nn.ModuleList([Block(D, H, F) for _ in range(L)]); self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
    def encode(self, ids):
        B, N = ids.shape; p = torch.arange(N, device=ids.device).unsqueeze(0).expand(B,-1)
        h = self.te(ids) + self.pe(p)
        for l in self.ly: h = l(h, causal=False)
        return self.norm(h)
    def get_embedding(self, h): return h[:, 0, :]
    def compute_loss(self, ids, mt, mp=0.15):
        B, N = ids.shape; mask = torch.rand(B,N,device=ids.device) < mp
        mi = ids.clone(); mi[mask] = mt; la = ids.clone(); la[~mask] = -100
        return F.cross_entropy(self.head(self.encode(mi)).reshape(-1, self.V), la.reshape(-1))
    def update_ema(self, m): pass


# ============================================================
# JEPA-LM (Base)
# ============================================================
class JEPALMModel(nn.Module):
    def __init__(self, V, D, L, H, F, M):
        super().__init__(); self.dim = D
        self.te = nn.Embedding(V, D); self.pe = nn.Embedding(M, D)
        self.ly = nn.ModuleList([Block(D, H, F) for _ in range(L)]); self.norm = nn.LayerNorm(D)
        self.tly = nn.ModuleList([Block(D, H, F) for _ in range(L)]); self.tnorm = nn.LayerNorm(D)
        for p in self.tly.parameters(): p.requires_grad = False
        for p in self.tnorm.parameters(): p.requires_grad = False
        pd = D//2; self.pp = nn.Linear(D, pd); self.po = nn.Linear(pd, D)
        self.mt = nn.Parameter(torch.randn(1,1,pd)*0.02)
        self.pl = nn.ModuleList([Block(pd, H, F//2) for _ in range(2)]); self.pnorm = nn.LayerNorm(pd)
        self._sync()
    @torch.no_grad()
    def _sync(self):
        self.tly.load_state_dict(self.ly.state_dict()); self.tnorm.load_state_dict(self.norm.state_dict())
    @torch.no_grad()
    def update_ema(self, m=0.996):
        for p, tp in zip(self.ly.parameters(), self.tly.parameters()): tp.data.mul_(m).add_(p.data, alpha=1-m)
        for p, tp in zip(self.norm.parameters(), self.tnorm.parameters()): tp.data.mul_(m).add_(p.data, alpha=1-m)
    def _enc(self, x, ly, nm):
        B,N = x.shape; p = torch.arange(N, device=x.device).unsqueeze(0).expand(B,-1)
        h = self.te(x) + self.pe(p)
        for l in ly: h = l(h, causal=False)
        return nm(h)
    def encode(self, ids): return self._enc(ids, self.ly, self.norm)
    @torch.no_grad()
    def encode_target(self, ids): return self._enc(ids, self.tly, self.tnorm)
    def get_embedding(self, h): return h.mean(dim=1)
    def predict(self, h, mp):
        B,N,D = h.shape; T = mp.shape[1]; c = self.pp(h); mtok = self.mt.expand(B,T,-1)
        ca = torch.gather(c, 1, mp.unsqueeze(-1).expand(-1,-1, c.size(-1)))
        x = torch.cat([ca, mtok], dim=1)
        for l in self.pl: x = l(x, causal=False)
        return self.po(self.pnorm(x)[:, T:, :])
    def compute_loss(self, ids, mt, mp=0.15):
        B,N = ids.shape; mask = torch.rand(B,N,device=ids.device) < mp
        mi = ids.clone(); mi[mask] = mt; h = self.encode(mi); th = self.encode_target(ids)
        mpl = [mask[b].nonzero(as_tuple=True)[0] for b in range(B) if mask[b].any()]
        if not mpl: return torch.tensor(0.0, device=ids.device, requires_grad=True)
        max_t = max(len(p) for p in mpl)
        pp = torch.zeros(len(mpl), max_t, dtype=torch.long, device=ids.device)
        vv = torch.zeros(len(mpl), max_t, dtype=torch.bool, device=ids.device)
        for b, pos in enumerate(mpl): pp[b,:len(pos)] = pos; vv[b,:len(pos)] = True
        pred = self.predict(h, pp)
        tat = torch.gather(th, 1, pp.unsqueeze(-1).expand(-1,-1, th.size(-1)))
        return (1.0 - F.cosine_similarity(pred, tat, dim=-1)[vv]).mean()
    def forward(self, ids): return self.encode(ids)


# ============================================================
# LLM-JEPA (Traditional)
# ============================================================
class LLMJEPA(nn.Module):
    def __init__(self, V, D, L, H, F, M):
        super().__init__(); self.dim = D
        self.te = nn.Embedding(V, D); self.pe = nn.Embedding(M, D)
        self.ly = nn.ModuleList([Block(D, H, F) for _ in range(L)]); self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V); self.head.weight = self.te.weight
        pd = D//2; self.pp = nn.Linear(D, pd); self.po = nn.Linear(pd, D)
        self.mt = nn.Parameter(torch.randn(1,1,pd)*0.02)
        self.pl = nn.ModuleList([Block(pd, H, F//2) for _ in range(2)]); self.pnorm = nn.LayerNorm(pd)
    def encode(self, ids):
        B,N = ids.shape; p = torch.arange(N, device=ids.device).unsqueeze(0).expand(B,-1)
        h = self.te(ids) + self.pe(p)
        for l in self.ly: h = l(h, causal=True)
        return self.norm(h)
    def get_embedding(self, h): return h[:, -1, :]
    def predict(self, h, mp):
        B,N,D = h.shape; T = mp.shape[1]; c = self.pp(h); mtok = self.mt.expand(B,T,-1)
        ca = torch.gather(c, 1, mp.unsqueeze(-1).expand(-1,-1, c.size(-1)))
        x = torch.cat([ca, mtok], dim=1)
        for l in self.pl: x = l(x, causal=False)
        return self.po(self.pnorm(x)[:, T:, :])
    def compute_loss(self, ids, mt, mp=0.15):
        B,N = ids.shape
        lo = self.forward(ids); t = ids[:,1:].contiguous(); lo = lo[:,:-1,:].contiguous()
        ntp = F.cross_entropy(lo.reshape(-1, lo.size(-1)), t.reshape(-1))
        mask = torch.rand(B,N,device=ids.device) < mp
        h = self.encode(ids.clone().masked_fill(mask, mt))
        mpl = [mask[b].nonzero(as_tuple=True)[0] for b in range(B) if mask[b].any()]
        jl = torch.tensor(0.0, device=ids.device, requires_grad=True)
        if mpl:
            max_t = max(len(p) for p in mpl)
            pp = torch.zeros(len(mpl), max_t, dtype=torch.long, device=ids.device)
            vv = torch.zeros(len(mpl), max_t, dtype=torch.bool, device=ids.device)
            for b, pos in enumerate(mpl): pp[b,:len(pos)] = pos; vv[b,:len(pos)] = True
            pred = self.predict(h, pp)
            th = self.encode(ids)
            tat = torch.gather(th, 1, pp.unsqueeze(-1).expand(-1,-1, th.size(-1)))
            jl = (1.0 - F.cosine_similarity(pred, tat, dim=-1)[vv]).mean()
        return 0.9*ntp + 0.1*jl
    def forward(self, ids): return self.head(self.encode(ids))
    def update_ema(self, m): pass


# ============================================================
# H-JEPA-LM (Our Improvement — Hierarchical + Action Conditioning)
# ============================================================
class HJEPALM(nn.Module):
    """Hierarchical JEPA-LM with Action Conditioning.
    
    Key improvements over base JEPA-LM:
    1. Multi-level prediction: low-level = token details, high-level = semantic meaning
    2. Action conditioning: can predict consequences of actions
    3. World model: plan actions in latent space
    """
    def __init__(self, V, D, L, H, F, M, num_levels=2, action_dim=32):
        super().__init__()
        self.dim = D; self.V = V; self.num_levels = num_levels
        self.te = nn.Embedding(V, D); self.pe = nn.Embedding(M, D)

        layers_per_level = max(1, L // num_levels)
        self.level_enc = nn.ModuleList()
        self.level_norm = nn.ModuleList()
        for _ in range(num_levels):
            self.level_enc.append(nn.ModuleList([Block(D, H, F) for _ in range(layers_per_level)]))
            self.level_norm.append(nn.LayerNorm(D))

        self.tly = nn.ModuleList()
        self.tnorm = nn.ModuleList()
        for _ in range(num_levels):
            tly = nn.ModuleList([Block(D, H, F) for _ in range(layers_per_level)])
            tnorm = nn.LayerNorm(D)
            for p in tly.parameters(): p.requires_grad = False
            for p in tnorm.parameters(): p.requires_grad = False
            self.tly.append(tly); self.tnorm.append(tnorm)

        pd = D // 2
        self.level_pred = nn.ModuleList()
        self.level_mtok = nn.ParameterList()
        for lvl in range(num_levels):
            self.level_pred.append(nn.ModuleDict({
                'proj': nn.Linear(D, pd),
                'out': nn.Linear(pd, D),
                'layers': nn.ModuleList([Block(pd, H, F//2) for _ in range(2)]),
                'norm': nn.LayerNorm(pd),
            }))
            self.level_mtok.append(nn.Parameter(torch.randn(1,1,pd)*0.02))

        self.act_enc = nn.Sequential(nn.Linear(action_dim, D), nn.GELU(), nn.Linear(D, D))
        self.act_combine = nn.Linear(D*2, D)
        self.level_weights = nn.Parameter(torch.ones(num_levels))

        self._sync_targets()

    @torch.no_grad()
    def _sync_targets(self):
        for lvl in range(self.num_levels):
            self.tly[lvl].load_state_dict(self.level_enc[lvl].state_dict())
            self.tnorm[lvl].load_state_dict(self.level_norm[lvl].state_dict())

    @torch.no_grad()
    def update_ema(self, m=0.996):
        for lvl in range(self.num_levels):
            for p, tp in zip(self.level_enc[lvl].parameters(), self.tly[lvl].parameters()):
                tp.data.mul_(m).add_(p.data, alpha=1-m)
            for p, tp in zip(self.level_norm[lvl].parameters(), self.tnorm[lvl].parameters()):
                tp.data.mul_(m).add_(p.data, alpha=1-m)

    def encode(self, ids):
        B, N = ids.shape; p = torch.arange(N, device=ids.device).unsqueeze(0).expand(B,-1)
        h = self.te(ids) + self.pe(p)
        level_outputs = []
        for lvl in range(self.num_levels):
            for l in self.level_enc[lvl]: h = l(h, causal=False)
            h = self.level_norm[lvl](h)
            level_outputs.append(h)
        return level_outputs

    @torch.no_grad()
    def encode_target(self, ids):
        B, N = ids.shape; p = torch.arange(N, device=ids.device).unsqueeze(0).expand(B,-1)
        h = self.te(ids) + self.pe(p)
        level_outputs = []
        for lvl in range(self.num_levels):
            for l in self.tly[lvl]: h = l(h, causal=False)
            h = self.tnorm[lvl](h)
            level_outputs.append(h)
        return level_outputs

    def get_embedding(self, level_outputs):
        return level_outputs[-1].mean(dim=1)

    def predict_level(self, h, mp, lvl, actions=None):
        B, N, D = h.shape; T = mp.shape[1]
        pd = self.level_pred[lvl]
        c = pd['proj'](h)
        mtok = self.level_mtok[lvl].expand(B, T, -1)
        ca = torch.gather(c, 1, mp.unsqueeze(-1).expand(-1,-1, c.size(-1)))

        if actions is not None and lvl == self.num_levels - 1:
            act = self.act_enc(actions).unsqueeze(1).expand(-1, T, -1)
            ca = self.act_combine(torch.cat([ca, act], dim=-1))

        x = torch.cat([ca, mtok], dim=1)
        for l in pd['layers']: x = l(x, causal=False)
        return pd['out'](pd['norm'](x)[:, T:, :])

    def compute_loss(self, ids, mt, actions=None, mp=0.15):
        B, N = ids.shape; mask = torch.rand(B, N, device=ids.device) < mp
        mi = ids.clone(); mi[mask] = mt

        level_enc = self.encode(mi)
        with torch.no_grad():
            level_tgt = self.encode_target(ids)

        mpl = [mask[b].nonzero(as_tuple=True)[0] for b in range(B) if mask[b].any()]
        if not mpl:
            return torch.tensor(0.0, device=ids.device, requires_grad=True)

        max_t = max(len(p) for p in mpl)
        pp = torch.zeros(len(mpl), max_t, dtype=torch.long, device=ids.device)
        vv = torch.zeros(len(mpl), max_t, dtype=torch.bool, device=ids.device)
        for b, pos in enumerate(mpl): pp[b,:len(pos)] = pos; vv[b,:len(pos)] = True

        weights = F.softmax(self.level_weights, dim=0)
        total_loss = torch.tensor(0.0, device=ids.device, requires_grad=True)

        for lvl in range(self.num_levels):
            pred = self.predict_level(level_enc[lvl], pp, lvl, actions)
            tat = torch.gather(level_tgt[lvl], 1, pp.unsqueeze(-1).expand(-1,-1, level_tgt[lvl].size(-1)))
            cos_sim = F.cosine_similarity(pred, tat, dim=-1)
            lvl_loss = (1.0 - cos_sim[vv]).mean()
            total_loss = total_loss + weights[lvl] * lvl_loss

        return total_loss

    def predict_next_state(self, ids, actions):
        with torch.no_grad():
            level_out = self.encode(ids)
            h = level_out[-1]
        act = self.act_enc(actions)
        return h[:, -1:, :] + act.unsqueeze(1)

    def plan_actions(self, ids, goal_latent, num_steps=3, num_candidates=5):
        with torch.no_grad():
            level_out = self.encode(ids)
            h = level_out[-1].clone()
        actions = []
        act_dim = self.act_enc[0].in_features
        for _ in range(num_steps):
            cands = torch.randn(ids.shape[0], num_candidates, act_dim, device=ids.device)
            best_d = float('inf'); best_a = None
            for c in range(num_candidates):
                a_emb = self.act_enc(cands[:, c, :])
                ns = h[:, -1:, :] + a_emb.unsqueeze(1)
                d = F.mse_loss(ns, goal_latent.unsqueeze(1), reduction='none').mean(dim=[1,2])
                if d.mean().item() < best_d: best_d = d.mean().item(); best_a = cands[:, c, :]
            a_emb = self.act_enc(best_a)
            h = h.clone(); h[:, -1:, :] = h[:, -1:, :] + a_emb.unsqueeze(1)
            actions.append(best_a.unsqueeze(1))
        return torch.cat(actions, dim=1)

    def forward(self, ids):
        return self.encode(ids)[-1]


# ============================================================
# UTILITIES
# ============================================================
class LinClf(nn.Module):
    def __init__(self, enc, nc):
        super().__init__(); self.enc = enc
        for p in self.enc.parameters(): p.requires_grad = False
        self.head = nn.Linear(enc.dim, nc)
    def forward(self, ids):
        h = self.enc.encode(ids)
        if isinstance(h, list): h = h[-1]
        return self.head(h.mean(dim=1))


class TokDS(Dataset):
    def __init__(self, tl): self.d = [torch.tensor(t, dtype=torch.long) for t in tl]
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]

class ClfDS(Dataset):
    def __init__(self, texts, labels, tok, ml):
        self.labels = labels; self.enc = tok(texts, truncation=True, padding="max_length", max_length=ml, return_tensors="pt")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {"input_ids": self.enc["input_ids"][i], "label": self.labels[i]}


def pretrain(model, tok, seqs, cfg, dev, epochs, mtype):
    is_jepa = mtype in ("JEPA-LM", "H-JEPA-LM")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.05)
    ds = TokDS(seqs); loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True, drop_last=True)
    sc = torch.amp.GradScaler("cuda"); t0 = time.time()
    for ep in range(epochs):
        model.train(); tl = 0; n = 0
        for batch in loader:
            ids = batch.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if mtype == "GPT-NTP": loss = model.compute_loss(ids)
                else: loss = model.compute_loss(ids, tok.mask_token_id)
            sc.scale(loss).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            sc.step(opt); sc.update(); opt.zero_grad()
            if is_jepa: model.update_ema(0.996)
            tl += loss.item(); n += 1
        print(f"    Ep {ep+1}/{epochs}: loss={tl/max(1,n):.4f} ({time.time()-t0:.0f}s)")


def eval_clf(enc, tok, tr, va, dev, nc=2, ep=3, ml=128):
    tds = ClfDS([x["text"] for x in tr], [x["label"] for x in tr], tok, ml)
    vds = ClfDS([x["text"] for x in va], [x["label"] for x in va], tok, ml)
    tl = DataLoader(tds, batch_size=128, shuffle=True); vl = DataLoader(vds, batch_size=128)
    clf = LinClf(enc, nc).to(dev); opt = torch.optim.Adam(clf.parameters(), lr=2e-4)
    for e in range(ep):
        clf.train()
        for b in tl:
            ids = b["input_ids"].to(dev); la = b["label"].to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(clf(ids), la)
            loss.backward(); opt.step(); opt.zero_grad()
    clf.eval(); co = 0; tt = 0
    with torch.no_grad():
        for b in vl:
            ids = b["input_ids"].to(dev); la = b["label"].to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16): lo = clf(ids)
            co += (lo.argmax(-1)==la).sum().item(); tt += la.size(0)
    return co/max(1,tt)


def get_emb(enc, ids):
    h = enc.encode(ids)
    if isinstance(h, list): h = h[-1]
    return h.mean(dim=1)


def eval_div(enc, tok, wiki, dev, ns=100):
    tx = [x["text"] for x in wiki[:ns]]
    ids = tok(tx, truncation=True, padding="max_length", max_length=128, return_tensors="pt")["input_ids"].to(dev)
    enc.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        em = get_emb(enc, ids).float()
    cs = F.cosine_similarity(em[:-1], em[1:], dim=-1).mean().item()
    std = em.std(dim=0).mean().item()
    _, S, _ = torch.linalg.svd(em.float(), full_matrices=False)
    sv = (S[0]/S.sum()).item() if S.sum() > 0 else 1.0
    return {"cosine_sim": cs, "embedding_std": std, "sv_ratio": sv}


def eval_reason(enc, tok, wiki, dev, V, M, ne=200):
    at = []
    for it in wiki:
        at.extend(tok.encode(it["text"], add_special_tokens=False))
        if len(at) > M*(ne+10): break
    pf, ta = [], []
    for i in range(0, min(len(at)-M, ne*2), M):
        pr = at[i:i+M-1]; tg = at[i+M-1]
        if len(pr)==M-1: pf.append(pr); ta.append(tg)
        if len(pf)>=ne: break
    if not pf: return {"top1":0, "top5":0}
    pt = torch.tensor(pf, dtype=torch.long, device=dev); tt = torch.tensor(ta, dtype=torch.long, device=dev)
    pr = nn.Linear(enc.dim, V).to(dev); opt = torch.optim.Adam(pr.parameters(), lr=5e-4)
    enc.eval()
    for e in range(3):
        pr.train()
        for i in range(0, len(pf), 64):
            ids = pt[i:i+64]; la = tt[i:i+64]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                em = get_emb(enc, ids).float()
            lo = pr(em); F.cross_entropy(lo, la).backward(); opt.step(); opt.zero_grad()
    pr.eval(); c1=c5=ttot=0
    with torch.no_grad():
        for i in range(0, len(pf), 64):
            ids = pt[i:i+64]; la = tt[i:i+64]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                em = get_emb(enc, ids).float()
            lo = pr(em); c1+=(lo.argmax(-1)==la).sum().item()
            t5=lo.topk(5,dim=-1).indices; c5+=(t5==la.unsqueeze(-1)).any(dim=-1).sum().item(); ttot+=la.size(0)
    return {"top1":c1/max(1,ttot), "top5":c5/max(1,ttot)}


def main():
    print("="*80)
    print("  H-JEPA-LM BENCHMARK: 5-WAY COMPARISON")
    print("  GPT-NTP vs BERT-MLM vs JEPA-LM vs LLM-JEPA vs H-JEPA-LM")
    print("="*80)
    print()
    print("  H-JEPA-LM = Hierarchical prediction + Action conditioning + World model")
    print("  (Our improvement over base JEPA-LM)")
    print()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {dev}")
    if dev.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    D, L, H, F, M, B = 128, 4, 4, 512, 128, 64
    cfg = {"dim":D, "layers":L, "heads":H, "ff_dim":F, "seq_len":M, "batch":B, "lr":3e-4}

    tok = AutoTokenizer.from_pretrained("bert-base-uncased"); V = len(tok); cfg["vocab_size"] = V

    print("\n  Loading Wikipedia (2000 articles)...")
    wiki_raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    wiki = [{"text":x["text"]} for x in wiki_raw if x["text"].strip()][:2000]
    print(f"    {len(wiki)} articles")

    print("  Loading SST-2...")
    sst_tr = [{"text":x["sentence"], "label":x["label"]} for x in load_dataset("glue","sst2",split="train")]
    sst_va = [{"text":x["sentence"], "label":x["label"]} for x in load_dataset("glue","sst2",split="validation")]

    print("  Tokenizing...")
    seqs = []
    for it in wiki:
        t = tok.encode(it["text"], add_special_tokens=False)
        for i in range(0, len(t)-M, M): seqs.append(t[i:i+M])
    print(f"    {len(seqs)} sequences")

    R = {}; E = 3
    models = [
        ("GPT-NTP", lambda: GPTModel(V, D, L, H, F, M)),
        ("BERT-MLM", lambda: BERTModel(V, D, L, H, F, M)),
        ("JEPA-LM", lambda: JEPALMModel(V, D, L, H, F, M)),
        ("LLM-JEPA", lambda: LLMJEPA(V, D, L, H, F, M)),
        ("H-JEPA-LM", lambda: HJEPALM(V, D, L, H, F, M, num_levels=2, action_dim=32)),
    ]

    print("\n" + "="*80)
    print("  TRAINING + EVALUATION")
    print("="*80)

    for name, factory in models:
        print(f"\n  --- {name} ---")
        enc = factory().to(dev)
        params = sum(p.numel() for p in enc.parameters())
        print(f"  Params: {params:,}")
        pretrain(enc, tok, seqs, cfg, dev, E, name)
        acc = eval_clf(enc, tok, sst_tr[:1000], sst_va, dev, ep=3, ml=M)
        R[name] = {"classification_acc": acc, "params": params}
        print(f"  SST-2: {acc:.4f}")
        dv = eval_div(enc, tok, wiki, dev)
        R[name]["diversity"] = dv
        print(f"  Diversity: cos={dv['cosine_sim']:.4f} std={dv['embedding_std']:.4f} sv={dv['sv_ratio']:.4f}")
        re = eval_reason(enc, tok, wiki, dev, V, M, ne=200)
        R[name]["reasoning_top1"] = re["top1"]; R[name]["reasoning_top5"] = re["top5"]
        print(f"  Reasoning: T1={re['top1']:.4f} T5={re['top5']:.4f}")
        torch.save(enc.state_dict(), os.path.join(RESULTS_DIR, f"{name}.pt"))
        del enc; torch.cuda.empty_cache()

    # SAMPLE EFFICIENCY
    print("\n" + "="*80)
    print("  SAMPLE EFFICIENCY")
    print("="*80)
    for name, factory in models:
        print(f"\n  --- {name} ---"); R[f"{name}_se"] = {}
        for frac in [0.1, 0.5, 1.0]:
            n = max(50, int(len(sst_tr)*frac)); sub = sst_tr[:n]
            fr = factory().to(dev)
            pretrain(fr, tok, seqs, cfg, dev, E, name)
            a = eval_clf(fr, tok, sub, sst_va, dev, ep=3, ml=M)
            R[f"{name}_se"][f"{int(frac*100)}%"] = a
            print(f"    {int(frac*100)}% data -> {a:.4f}")
            del fr; torch.cuda.empty_cache()

    # RESULTS
    print("\n" + "="*80)
    print("  FINAL RESULTS")
    print("="*80)
    MN = ["GPT-NTP", "BERT-MLM", "JEPA-LM", "LLM-JEPA", "H-JEPA-LM"]
    print(f"\n{'Metric':<35} {'GPT':>8} {'BERT':>8} {'JEPA':>8} {'LLM-J':>8} {'H-JEPA':>8}  Winner")
    print("-"*95)
    for m, k, hb in [
        ("SST-2 Accuracy", "classification_acc", True),
        ("Reasoning Top-1", "reasoning_top1", True),
        ("Reasoning Top-5", "reasoning_top5", True),
        ("Cosine Sim (lower=better)", ("diversity","cosine_sim"), False),
        ("Embed Std (higher=better)", ("diversity","embedding_std"), True),
        ("SV Ratio (lower=better)", ("diversity","sv_ratio"), False),
    ]:
        vs = {}
        for n in MN:
            v = R.get(n, {})
            vs[n] = v.get(k[0],{}).get(k[1],0) if isinstance(k, tuple) else v.get(k, 0)
        bv = max(vs.values()) if hb else min(vs.values())
        w = [n for n,v in vs.items() if abs(v-bv)<1e-6 and v!=0]
        wn = "/".join(w) if w else "-"
        print(f"{m:<35} {vs['GPT-NTP']:>8.4f} {vs['BERT-MLM']:>8.4f} {vs['JEPA-LM']:>8.4f} {vs['LLM-JEPA']:>8.4f} {vs['H-JEPA-LM']:>8.4f}  <- {wn}")

    print(f"\n{'Sample Efficiency':<35}")
    for f in ["10%", "50%", "100%"]:
        vs = {}
        for n in MN: vs[n] = R.get(f"{n}_se",{}).get(f,0)
        bv = max(vs.values()); w = [n for n,v in vs.items() if abs(v-bv)<1e-6 and v!=0]
        wn = "/".join(w) if w else "-"
        print(f"  {f+' data':<33} {vs['GPT-NTP']:>8.4f} {vs['BERT-MLM']:>8.4f} {vs['JEPA-LM']:>8.4f} {vs['LLM-JEPA']:>8.4f} {vs['H-JEPA-LM']:>8.4f}  <- {wn}")

    # VERDICT
    print("\n" + "="*80)
    print("  VERDICT")
    print("="*80)
    print()
    print("  Evolution of JEPA for Text:")
    print("    LLM-JEPA (ICLR 2026) -> JEPA as secondary, causal, no EMA")
    print("    JEPA-LM (Ours v1)    -> JEPA as primary, bidirectional, EMA")
    print("    H-JEPA-LM (Ours v2)  -> H-JEPA + Action conditioning + World model")
    print()

    hj = R.get("H-JEPA-LM", {})
    jl = R.get("JEPA-LM", {})
    ll = R.get("LLM-JEPA", {})
    print(f"  H-JEPA-LM vs Base JEPA-LM:")
    print(f"    Embedding diversity: {hj.get('diversity',{}).get('cosine_sim',0):.4f} vs {jl.get('diversity',{}).get('cosine_sim',0):.4f}")
    print(f"    Classification:      {hj.get('classification_acc',0):.4f} vs {jl.get('classification_acc',0):.4f}")
    print()
    print(f"  H-JEPA-LM vs LLM-JEPA (traditional):")
    print(f"    Embedding diversity: {hj.get('diversity',{}).get('cosine_sim',0):.4f} vs {ll.get('diversity',{}).get('cosine_sim',0):.4f}")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
