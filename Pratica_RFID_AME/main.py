# Sistema RFID + Deep Learning — Raspberry Pi Pico 2W
# Adaptado para UART (Terminal) e Buzzer
from machine import Pin, SPI, PWM
import time, json

from rfid_rc522 import MFRC522
from model      import classify, LABELS

# ── Pinos configurados por você ─────────────────────────────
# SPI0 usando SCK=GP2, MOSI=GP3, MISO=GP4
spi = SPI(0, baudrate=1_000_000, polarity=0, phase=0,
          sck=Pin(2), mosi=Pin(3), miso=Pin(4))

# CS=GP5, RST=GP0
rfid = MFRC522(spi=spi, gpioRst=Pin(0), gpioCs=Pin(5))

# Buzzer Ativo = GP18
buzzer = PWM(Pin(18))

# ── Banco de UIDs cadastrados ─────────────────────────────────
DB_FILE = "rfid_db.json"

def db_load():
    try:
        with open(DB_FILE) as f:
            return json.load(f)
    except:
        return {}

def db_save(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

db = db_load()

# ── Histórico temporal por cartão ─────────────────────────────
history = {}
WINDOW = 8

def update_history(uid_hex, uid_hash):
    import time as _t
    now   = _t.localtime()
    hora  = (now[3] * 3600 + now[4] * 60 + now[5])
    dia   = now[6]

    if uid_hex not in history:
        history[uid_hex] = []

    hist = history[uid_hex]
    if hist:
        last_t = hist[-1][2] * 86400
        delta  = max(0, hora - last_t) / 86400
    else:
        delta  = 0.5

    feat = [hora / 86400, dia / 7, delta, uid_hash / 255]
    hist.append(feat)
    if len(hist) > WINDOW:
        hist.pop(0)

    while len(hist) < WINDOW:
        hist.insert(0, feat)

    return hist[-WINDOW:]

# ── Helpers de hardware (Buzzer) ──────────────────────────────
def beep(freq=1500, ms=150):
    buzzer.freq(freq)
    buzzer.duty_u16(32768)
    time.sleep_ms(ms)
    buzzer.duty_u16(0)

def beep_alert():
    for f in [2000, 1500, 2000]:
        beep(freq=f, ms=120)
        time.sleep_ms(60)

def uid_to_hex(uid_bytes):
    return "".join(f"{b:02X}" for b in uid_bytes)

# ── Inicialização e Modo de Cadastro Automático ───────────────
print("=========================================")
print("  SISTEMA INICIADO - MODO CADASTRO (10s) ")
print("  Aproxime cartoes novos agora!          ")
print("=========================================")
beep(1000, 500)

deadline = time.ticks_add(time.ticks_ms(), 10_000)
while time.ticks_diff(deadline, time.ticks_ms()) > 0:
    stat, tag_type = rfid.request(rfid.REQIDL)
    if stat == rfid.OK:
        stat, raw_uid = rfid.anticoll()
        if stat == rfid.OK:
            uid_hex  = uid_to_hex(raw_uid)
            uid_hash = sum(raw_uid) % 256
            if uid_hex not in db:
                db[uid_hex] = {"nome": f"User_{uid_hex[:4]}", "hash": uid_hash}
                db_save(db)
                print(f"[+] NOVO CARTAO CADASTRADO: {uid_hex}")
                beep(freq=1200, ms=300)
            time.sleep(1)

print("\n=========================================")
print("  MODO DE CONTROLE DE ACESSO INICIADO    ")
print("=========================================")
beep(1500, 100)
time.sleep_ms(100)
beep(1500, 100)

# ── Loop principal ─────────────────────────────────────────────
while True:
    stat, tag_type = rfid.request(rfid.REQIDL)
    if stat != rfid.OK:
        time.sleep_ms(100)
        continue

    stat, raw_uid = rfid.anticoll()
    if stat != rfid.OK:
        continue

    uid_hex  = uid_to_hex(raw_uid)
    uid_hash = sum(raw_uid) % 256
    t_start  = time.ticks_ms()

    import time as t
    lt = t.localtime()
    ts = f"{lt[3]:02d}:{lt[4]:02d}:{lt[5]:02d}"

    # 1. Cartão não cadastrado
    if uid_hex not in db:
        print(f"[{ts}] UID={uid_hex}  ACESSO NEGADO (Desconhecido)")
        beep_alert()
        time.sleep(2)
        continue

    # 2. Cartão Cadastrado -> Inferência na IA
    features = update_history(uid_hex, uid_hash)
    label, conf = classify(features)
    cls_idx = LABELS.index(label)
    t_ms = time.ticks_diff(time.ticks_ms(), t_start)
    nome = db[uid_hex]["nome"]

    # 3. Tratamento dos Resultados
    if cls_idx == 0:  # NORMAL
        print(f"[{ts}] UID={uid_hex} ({nome}) -> LIBERADO  [Conf:{conf}%] [Tempo:{t_ms}ms]")
        beep(freq=1200, ms=100)
    
    elif cls_idx == 1: # SUSPEITO / ATÍPICO
        print(f"[{ts}] UID={uid_hex} ({nome}) -> ATIPICO!  [Conf:{conf}%] [Tempo:{t_ms}ms]")
        beep(freq=800, ms=300)
    
    else: # BLOQUEADO / ANOMALIA GRAVE
        print(f"[{ts}] UID={uid_hex} ({nome}) -> BLOQUEADO PELO MODELO [Conf:{conf}%] [Tempo:{t_ms}ms]")
        beep_alert()

    time.sleep(2)