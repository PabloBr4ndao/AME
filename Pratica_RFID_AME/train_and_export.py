"""
src/model/train_and_export.py
─────────────────────────────
Pipeline Deep Learning para sistema RFID:
  1. Gera dataset sintético de acessos temporais
  2. Treina LSTM(16)×2 → Dense(8,ReLU) → Dense(3,Softmax)
  3. Poda por magnitude (|w| < 0.05)
  4. Quantiza float32 → int8
  5. Exporta model.py para a Raspberry Pi Pico 2W

Executar no PC:  python src/model/train_and_export.py
Requer:          pip install numpy
"""

import numpy as np
import random, math
from pathlib import Path

random.seed(42)
np.random.seed(42)

SEQ_LEN   = 8      # janela temporal (últimos N acessos)
INPUT_DIM = 4      # [hora/24, dia/7, delta_t, uid_hash/255]
HIDDEN    = 16     # unidades LSTM por camada
DENSE_H   = 8      # neurônios da camada densa
N_CLASSES = 3      # NORMAL=0, SUSPEITO=1, BLOQUEADO=2
EPOCHS    = 2000
LR        = 0.005
PRUNE_TH  = 0.05
OUTPUT_DIR = Path(__file__).parent


# ════════════════════════════════════════════════════════
# 1. GERAÇÃO DO DATASET
# ════════════════════════════════════════════════════════

def gen_sequence(uid_hash, label):
    """
    Gera uma janela de SEQ_LEN acessos.
    label 0 = NORMAL   : horário comercial, intervalo regular
    label 1 = SUSPEITO : inicia normal, vira noturno
    label 2 = BLOQUEADO: rajada rápida (ataque/clone)
    """
    seq = []
    if label == 0:
        base = random.uniform(7, 18)
        for _ in range(SEQ_LEN):
            h  = float(np.clip(base + random.gauss(0, 0.5), 0, 23)) / 24
            d  = random.randint(1, 5) / 7
            dt = random.uniform(3600, 86400) / 86400
            seq.append([h, d, dt, uid_hash / 255])

    elif label == 1:
        for i in range(SEQ_LEN):
            if i < SEQ_LEN // 2:
                h = float(np.clip(random.uniform(8, 17), 0, 23)) / 24
                d = random.randint(1, 5) / 7
            else:
                h = random.choice([random.uniform(0, 6),
                                   random.uniform(22, 24)]) / 24
                d = random.randint(0, 6) / 7
            dt = random.uniform(60, 7200) / 86400
            seq.append([h, d, dt, uid_hash / 255])

    else:
        hb = random.uniform(0, 24)
        for _ in range(SEQ_LEN):
            h  = float(np.clip(hb + random.gauss(0, 0.1), 0, 23)) / 24
            d  = random.randint(0, 6) / 7
            dt = random.uniform(1, 30) / 86400      # < 30 s entre tentativas
            seq.append([h, d, dt, uid_hash / 255])

    return np.array(seq, dtype=np.float32)


def make_dataset(n_per_class=200):
    uids = [random.randint(0, 255) for _ in range(10)]
    X, y = [], []
    for _ in range(n_per_class):
        for label in range(N_CLASSES):
            X.append(gen_sequence(random.choice(uids), label))
            y.append(label)
    return np.array(X), np.array(y)


print("=" * 55)
print("  RFID Deep Learning — Pico 2W + Freenove")
print("=" * 55)
X_train, y_train = make_dataset(200)
X_val,   y_val   = make_dataset(50)
print(f"\n[1/4] Dataset: {len(X_train)} treino / {len(y_val)} validação")


# ════════════════════════════════════════════════════════
# 2. REDE NEURAL: LSTM(16)×2 + Dense(8) + Dense(3)
# ════════════════════════════════════════════════════════

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def _softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def lstm_forward(X_seq, W, b):
    """
    X_seq : (batch, seq_len, input_dim)
    W     : (4*H, input_dim+H) — portas i,f,o,g concatenadas
    b     : (4*H,)
    """
    B, T, D = X_seq.shape
    H = W.shape[0] // 4
    h = np.zeros((B, H), dtype=np.float32)
    c = np.zeros((B, H), dtype=np.float32)
    for t in range(T):
        xh   = np.concatenate([X_seq[:, t, :], h], axis=1)
        gate = xh @ W.T + b
        i_g  = _sigmoid(gate[:, :H])
        f_g  = _sigmoid(gate[:, H:2*H])
        o_g  = _sigmoid(gate[:, 2*H:3*H])
        g_g  = np.tanh(np.clip(gate[:, 3*H:], -20, 20))
        c    = f_g * c + i_g * g_g
        h    = o_g * np.tanh(np.clip(c, -20, 20))
    return h


# Inicialização dos pesos
def _w(rows, cols):
    return (np.random.randn(rows, cols) * np.sqrt(2.0 / cols)).astype(np.float32)

W1 = _w(4*HIDDEN, INPUT_DIM + HIDDEN);  b1 = np.zeros(4*HIDDEN, dtype=np.float32)
W2 = _w(4*HIDDEN, HIDDEN    + HIDDEN);  b2 = np.zeros(4*HIDDEN, dtype=np.float32)
Wd = _w(HIDDEN,   DENSE_H);             bd = np.zeros(DENSE_H,  dtype=np.float32)
Wo = _w(DENSE_H,  N_CLASSES);           bo = np.zeros(N_CLASSES, dtype=np.float32)


def forward(X):
    h1 = lstm_forward(X, W1, b1)
    # Segunda LSTM recebe h1 repetido como sequência
    X2 = np.repeat(h1[:, np.newaxis, :], SEQ_LEN, axis=1)
    h2 = lstm_forward(X2, W2, b2)
    d1 = np.maximum(0, h2 @ Wd + bd)          # ReLU
    return _softmax(d1 @ Wo + bo)              # Softmax


def accuracy(X, y):
    return (forward(X).argmax(axis=1) == y).mean()


# ─── Treino SGD com gradiente da camada de saída ─────────────
print("\n[2/4] Treinamento...")
BATCH = 32
for ep in range(1, EPOCHS + 1):
    perm = np.random.permutation(len(X_train))
    for s in range(0, len(X_train), BATCH):
        idx = perm[s:s+BATCH]
        xb, yb = X_train[idx], y_train[idx]
        out = forward(xb)
        n   = len(yb)
        oh  = np.zeros_like(out)
        oh[np.arange(n), yb] = 1
        d   = (out - oh) / n
        # Atualiza apenas camadas densas (backprop simplificado)
        h1  = lstm_forward(xb, W1, b1)
        X2  = np.repeat(h1[:, np.newaxis, :], SEQ_LEN, axis=1)
        h2  = lstm_forward(X2, W2, b2)
        d1  = np.maximum(0, h2 @ Wd + bd)
        Wo -= LR * np.clip(d1.T @ d, -1, 1)
        bo -= LR * np.clip(d.sum(0), -1, 1)
        d_d1 = (d @ Wo.T) * (d1 > 0)
        Wd -= LR * np.clip(h2.T @ d_d1, -1, 1)
        bd -= LR * np.clip(d_d1.sum(0), -1, 1)

    if ep % 400 == 0:
        at = accuracy(X_train, y_train)
        av = accuracy(X_val, y_val)
        print(f"  Época {ep:4d}  treino={at:.1%}  val={av:.1%}")

print(f"  Acurácia final (val): {accuracy(X_val, y_val):.1%}")


# ════════════════════════════════════════════════════════
# 3. PODA
# ════════════════════════════════════════════════════════
print(f"\n[3/4] Poda (|w| < {PRUNE_TH})...")
all_W = [W1, b1, W2, b2, Wd, bd, Wo, bo]
total = sum(w.size for w in all_W)
for w in all_W:
    w *= (np.abs(w) >= PRUNE_TH)
zerados = sum((w == 0).sum() for w in all_W)
print(f"  Pesos zerados: {zerados}/{total} ({zerados/total:.1%})")
print(f"  Acurácia pós-poda (val): {accuracy(X_val, y_val):.1%}")


# ════════════════════════════════════════════════════════
# 4. QUANTIZAÇÃO E EXPORT
# ════════════════════════════════════════════════════════
print("\n[4/4] Quantização float32 → int8 + export...")

def quant(W):
    mv = float(np.max(np.abs(W)))
    if mv == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    sc = mv / 127.0
    return np.clip(np.round(W / sc), -128, 127).astype(np.int8), sc


W1q, sW1 = quant(W1); b1q, sb1 = quant(b1)
W2q, sW2 = quant(W2); b2q, sb2 = quant(b2)
Wdq, sWd = quant(Wd); bdq, sbd = quant(bd)
Woq, sWo = quant(Wo); boq, sbo = quant(bo)
print(f"  Memória: {total*4} B (float32) → {total} B (int8)  ({4}× menor)")


def fl(a): return a.flatten().tolist()

# Trocado o comentário interno de predict para aspas simples (''') para evitar conflito de string
model_py = f"""# model.py — AUTO-GERADO — NAO EDITAR
# RFID Deep Learning: LSTM({HIDDEN})x2 + Dense({DENSE_H}) + Dense({N_CLASSES}) int8
# Upload: mpremote cp src/model/model.py :model.py
import math
SW1={sW1:.10f}; SB1={sb1:.10f}
SW2={sW2:.10f}; SB2={sb2:.10f}
SWD={sWd:.10f}; SBD={sbd:.10f}
SWO={sWo:.10f}; SBO={sbo:.10f}
W1={fl(W1q)}
B1={fl(b1q)}
W2={fl(W2q)}
B2={fl(b2q)}
WD={fl(Wdq)}
BD={fl(bdq)}
WO={fl(Woq)}
BO={fl(boq)}
H={HIDDEN}; IN={INPUT_DIM}; DH={DENSE_H}; NC={N_CLASSES}; SL={SEQ_LEN}

def _sig(x): return 1.0/(1.0+math.exp(-max(-20.0,min(20.0,x))))
def _tanh(x): return math.tanh(max(-20.0,min(20.0,x)))
def _relu(x): return x if x>0 else 0.0

def _lstm_seq(seq, W, B, sW, sB):
    h=[0.0]*H; c=[0.0]*H
    for x in seq:
        xh=x+h
        g=[sum(W[(gi*H+j)*(IN+H)+i]*xh[i] for i in range(len(xh)))*sW+B[gi*H+j]*sB
           for gi in range(4) for j in range(H)]
        nh=[]; nc=[]
        for j in range(H):
            iv=_sig(g[j]); fv=_sig(g[H+j]); ov=_sig(g[2*H+j]); gv=_tanh(g[3*H+j])
            cv=fv*c[j]+iv*gv; nc.append(cv); nh.append(ov*_tanh(cv))
        h=nh; c=nc
    return h

def predict(features):
    '''
    features: lista de SL timesteps, cada um com IN floats
              Ex: [[hora/24, dia/7, delta_t, uid_hash/255], ...]
    Retorna:  [prob_normal, prob_suspeito, prob_bloqueado]
    '''
    h1 = _lstm_seq(features, W1, B1, SW1, SB1)
    h2 = _lstm_seq([h1]*SL,  W2, B2, SW2, SB2)
    d1 = [_relu(sum(WD[k*H+j]*h2[j] for j in range(H))*SWD+BD[k]*SBD) for k in range(DH)]
    z  = [sum(WO[k*DH+j]*d1[j] for j in range(DH))*SWO+BO[k]*SBO for k in range(NC)]
    zm=max(z); ex=[math.exp(v-zm) for v in z]; s=sum(ex)
    return [e/s for e in ex]

LABELS=["NORMAL","SUSPEITO","BLOQUEADO"]

def classify(features):
    p=predict(features); i=p.index(max(p))
    return LABELS[i], round(p[i]*100)
"""

(OUTPUT_DIR / "model.py").write_text(model_py)
print(f"  Salvo: {OUTPUT_DIR}/model.py ({len(model_py):,} bytes)")
print("\n" + "="*55)
print("  CONCLUIDO!")
print("  mpremote cp src/model/model.py :model.py")
print("="*55)