# src/hardware/rfid_rc522.py
# Driver MFRC522 para MicroPython (Raspberry Pi Pico 2W)
# Protocolo SPI 3.3V — compatível com módulo do kit Freenove
# Baseado na especificação MFRC522 da NXP

from machine import SPI, Pin
import time

class MFRC522:
    OK    = 0
    NOTAG = 1
    ERR   = 2

    REQIDL  = 0x26
    REQALL  = 0x52
    AUTHENT1A = 0x60

    # Registradores principais
    _CommandReg  = 0x01
    _ComIEnReg   = 0x02
    _ComIrqReg   = 0x04
    _ErrorReg    = 0x06
    _FIFODataReg = 0x09
    _FIFOLevelReg= 0x0A
    _ControlReg  = 0x0C
    _BitFramingReg=0x0D
    _ModeReg     = 0x11
    _TxControlReg= 0x14
    _TxASKReg    = 0x15
    _CRCResultReg= 0x21
    _TModeReg    = 0x2A
    _TPrescalerReg=0x2B
    _TReloadRegH = 0x2C
    _TReloadRegL = 0x2D
    _VersionReg  = 0x37

    def __init__(self, spi: SPI, gpioRst: Pin, gpioCs: Pin):
        self.spi = spi
        self.rst = gpioRst
        self.cs  = gpioCs
        self.rst.init(Pin.OUT, value=1)
        self.cs.init(Pin.OUT, value=1)
        self._reset()
        self._init()

    def _reset(self):
        self.rst.value(0)
        time.sleep_ms(50)
        self.rst.value(1)
        time.sleep_ms(50)

    def _write(self, reg, val):
        self.cs.value(0)
        self.spi.write(bytes([(reg << 1) & 0x7E, val]))
        self.cs.value(1)

    def _read(self, reg):
        self.cs.value(0)
        buf = bytearray(2)
        self.spi.write_readinto(bytes([((reg << 1) & 0x7E) | 0x80, 0]), buf)
        self.cs.value(1)
        return buf[1]

    def _set_bit(self, reg, mask):
        self._write(reg, self._read(reg) | mask)

    def _clr_bit(self, reg, mask):
        self._write(reg, self._read(reg) & (~mask))

    def _init(self):
        self._write(self._TModeReg,      0x8D)
        self._write(self._TPrescalerReg, 0x3E)
        self._write(self._TReloadRegL,   30)
        self._write(self._TReloadRegH,   0)
        self._write(self._TxASKReg,      0x40)
        self._write(self._ModeReg,       0x3D)
        self._set_bit(self._TxControlReg, 0x03)

    def _card_write(self, cmd, data):
        irq_en = 0x77 if cmd == 0x0C else 0x12
        self._write(self._ComIEnReg,  irq_en | 0x80)
        self._clr_bit(self._ComIrqReg, 0x80)
        self._set_bit(self._FIFOLevelReg, 0x80)
        self._write(self._CommandReg, 0x00)
        for b in data:
            self._write(self._FIFODataReg, b)
        self._write(self._CommandReg, cmd)
        if cmd == 0x0C:
            self._set_bit(self._BitFramingReg, 0x80)

        i = 2000
        while i > 0:
            n = self._read(self._ComIrqReg)
            i -= 1
            if n & 0x01 or n & irq_en:
                break
        self._clr_bit(self._BitFramingReg, 0x80)

        if i == 0:
            return self.ERR, None
        if self._read(self._ErrorReg) & 0x1B:
            return self.ERR, None

        back = []
        n = self._read(self._FIFOLevelReg)
        for _ in range(n):
            back.append(self._read(self._FIFODataReg))
        return self.OK, back

    def request(self, req_mode):
        self._write(self._BitFramingReg, 0x07)
        stat, back = self._card_write(0x0C, [req_mode])
        if stat != self.OK or len(back) != 2:
            return self.NOTAG, None
        return self.OK, back

    def anticoll(self):
        self._write(self._BitFramingReg, 0x00)
        stat, back = self._card_write(0x0C, [0x93, 0x20])
        if stat == self.OK and len(back) == 5:
            chk = 0
            for b in back[:4]:
                chk ^= b
            if chk != back[4]:
                return self.ERR, None
            return self.OK, back[:4]
        return self.ERR, None

    def version(self):
        return self._read(self._VersionReg)
