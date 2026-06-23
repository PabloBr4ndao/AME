

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/model"))

import random, math

# Importa o modelo gerado
try:
    from model import classify, predict, LABELS
except ImportError:
    print("ERRO: rode primeiro: python src/model/train_and_export.py")
    sys.exit(1)

random.seed(0)

def make_normal():
    """Janela de acesso em horário comercial."""
    return [[random.uniform(0.3, 0.7),  # hora 7h-17h
             random.randint(1,5)/7,
             random.uniform(0.04, 0.9),
             random.randint(50,200)/255]
            for _ in range(8)]

def make_suspeito():
    """Começa normal, termina noturno."""
    feat = []
    for i in range(8):
        if i < 4:
            feat.append([random.uniform(0.3,0.7), random.randint(1,5)/7,
                         random.uniform(0.04,0.9), 0.5])
        else:
            feat.append([random.uniform(0.0,0.1), random.randint(0,6)/7,
                         random.uniform(0.04,0.5), 0.5])
    return feat

def make_bloqueado():
    """Rajada de tentativas em < 30s."""
    return [[random.uniform(0,1), random.randint(0,6)/7,
             random.uniform(0.00001, 0.0004), 0.5]
            for _ in range(8)]

print("Testando model.py quantizado...")
N = 50
correct = 0
for _ in range(N):
    for fn, expected in [(make_normal, "NORMAL"),
                          (make_suspeito, "SUSPEITO"),
                          (make_bloqueado, "BLOQUEADO")]:
        label, conf = classify(fn())
        if label == expected:
            correct += 1

total = N * 3
acc = correct / total
print(f"Acurácia nos testes: {acc:.1%}  ({correct}/{total})")
assert acc >= 0.80, f"Acurácia insuficiente: {acc:.1%} < 80%"
print("PASSOU — model.py esta correto.")