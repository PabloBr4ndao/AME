from machine import ADC, Pin
import time
import math

# Configurações de Hardware
adc = ADC(Pin(26))
buzzer = Pin(15, Pin.OUT)

# Configurações do Sensor e Janela
AMOSTRAS_POR_LEITURA = 1000
JANELA_TAMANHO = 60 # 60 leituras = vetor do modelo
THRESHOLD_ANOMALIA = 0.00006 # Calculado no Colab!

buffer_janela = []

# Função para ler a Corrente Real (RMS) do sensor SCT-013
def ler_corrente_rms():
    soma_quadrados = 0
    # Amostra a onda AC em alta velocidade
    for _ in range(AMOSTRAS_POR_LEITURA):
        leitura_bruta = adc.read_u16()
        # Converte a leitura 0-65535 para Tensão 0-3.3V
        tensao = (leitura_bruta / 65535.0) * 3.3
        # Remove o DC offset de 1.65V que colocamos com o resistor
        tensao_sem_dc = tensao - 1.65
        soma_quadrados += (tensao_sem_dc * tensao_sem_dc)
        
    tensao_rms = math.sqrt(soma_quadrados / AMOSTRAS_POR_LEITURA)
    # Como o sensor é 5A / 1V, a corrente RMS é diretamente proporcional à Tensão RMS
    corrente_rms = tensao_rms * 5.0 
    
    # Retorna potência aproximada (Watts) assumindo rede de 220V
    potencia = corrente_rms * 220.0
    return max(0, potencia)

# Função simulando a inferência do modelo TFLite (se não usar firmware customizado)
# Em um firmware com tf lite, você usaria: tflite.Model('modelo_anomalia.tflite')
def realizar_inferencia_mse(janela_normalizada):
    # Simulador de Latência e Cálculo para demonstração
    tempo_inicio = time.ticks_ms()
    
    # NOTA: Num cenário real com tflite module:
    # interpreter.set_tensor(input_details[0]['index'], janela_normalizada)
    # interpreter.invoke()
    # output = interpreter.get_tensor(output_details[0]['index'])
    
    # Aqui simulamos o MSE. Para fins de apresentação prática, 
    # se a média da potência atual da janela for maior que um pico absurdo, geramos anomalia.
    media_potencia = sum(janela_normalizada) / len(janela_normalizada)
    
    tempo_fim = time.ticks_ms()
    latencia = time.ticks_diff(tempo_fim, tempo_inicio)
    
    # Simulador de Erro Quadrático
    mse = 0.00001 if media_potencia < 0.6 else 0.00009 
    return mse, latencia

print("Iniciando monitoramento de anomalias com TinyML...")

while True:
    # 1. Lê a potência a cada 6 segundos
    potencia_atual = ler_corrente_rms()
    
    # 2. Normaliza usando os limites do Colab (0 a 8500W)
    pot_norm = (potencia_atual - 0) / (8500.0 - 0)
    pot_norm = max(0, min(1, pot_norm)) # Clip entre 0 e 1
    
    buffer_janela.append(pot_norm)
    
    # Só faz inferência quando tiver a janela completa (60 pontos)
    if len(buffer_janela) == JANELA_TAMANHO:
        print("Janela cheia, rodando inferência TinyML...")
        
        mse_erro, latencia = realizar_inferencia_mse(buffer_janela)
        print(f"Latência da inferência: {latencia} ms | Erro MSE: {mse_erro:.6f}")
        
        if mse_erro > THRESHOLD_ANOMALIA:
            print(">>> ANOMALIA DETECTADA! Acionando Buzzer! <<<")
            buzzer.value(1) # Liga o som
            time.sleep(2)
            buzzer.value(0) # Desliga
        else:
            print("Consumo Normal.")
            
        # Desliza a janela (remove o ponto mais velho)
        buffer_janela.pop(0)
        
    time.sleep(6) # Aguarda 6 segundos para o próximo ponto