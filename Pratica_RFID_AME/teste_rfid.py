from machine import Pin, SPI
import time
from rfid_rc522 import MFRC522

print("Iniciando SPI...")
spi = SPI(0, baudrate=1_000_000, polarity=0, phase=0,
          sck=Pin(2), mosi=Pin(3), miso=Pin(4))

rfid = MFRC522(spi=spi, gpioRst=Pin(0), gpioCs=Pin(5))
print("Leitor RFID configurado! Aproxime um cartao...")

while True:
    stat, tag_type = rfid.request(rfid.REQIDL)
    if stat == rfid.OK:
        stat, raw_uid = rfid.anticoll()
        if stat == rfid.OK:
            uid_hex = "".join(f"{b:02X}" for b in raw_uid)
            print(f"CARTAO DETECTADO! UID: {uid_hex}")
            time.sleep(1) # Aguarda 1 segundo para não floodar a tela