# Detecção de Anomalias no Consumo de Energia Residencial (TinyML) ⚡

Este repositório contém o código-fonte final do projeto de detecção local e autônoma de anomalias elétricas, utilizando Aprendizado de Máquina Embarcado em uma **Raspberry Pi Pico 2 W**. 

🎓 **Disciplina:** Aprendizado de Máquina Embarcado (AME) - UFC Quixadá  
👨‍💻 **Autores:** Pablo Brandão Passos & Ronald M. Viana Lopes  
👨‍🏫 **Professor:** Jeandro de Mesquita Bezerra

## 📁 Estrutura do Repositório

```
/
├── README.md                           # Documentação e instruções de execução do projeto
├── requirements.txt                    # Dependências Python para treinar o modelo computacional
├── notebook/              
│   └── energy_anomaly_detection.ipynb  # Notebook principal (Treino e Quantização)
├── model/                     
│   └── modelo_anomalia_quant.tflite    # Modelo TinyML quantizado em INT8 (Apenas 4,5 KB)
├── pico/                         
│   └── main.py                         # Código MicroPython para rodar na Raspberry Pi Pico 2 W
└── data/                         
    └── link_dataset.md                 # Informações, estatísticas e link para o dataset original UK-DALE
```

## 🛠️ Hardware e Circuito Físico

O sistema processa as leituras fisicamente na borda extrema (Edge AI), garantindo latência próxima de zero e total privacidade de dados (sem envio para nuvem).

* **Microcontrolador:** Raspberry Pi Pico 2 W (Chip RP2040, 264KB RAM).
* **Sensor de Corrente:** SCT-013 (Modelo 5A / 1V) Não Invasivo.
* **Atuador:** Buzzer sonoro.
* **Circuito de Adequação AC:** Como o sensor SCT-013 gera corrente alternada (com tensões negativas), foi implementado um **Divisor de Tensão Capacitivo**:
  * 2x Resistores (Ex: 10kΩ) ligados entre o 3.3V(OUT) e o GND para criar um ponto médio de 1.65V.
  * 1x Capacitor (Ex: 10µF) para estabilização.
  * O sinal do sensor entra neste ponto médio e é lido pela porta analógica **GP26 (ADC0)**.


## 🚀 Como Executar o Projeto

### Parte 1: Treinar e Quantizar o Modelo (No Computador)
1. Clone este repositório no seu VS Code.
2. Crie e ative um ambiente virtual Python, e instale as dependências:
   `pip install -r requirements.txt`
3. Abra a pasta `notebook/`, execute o `energy_anomaly_detection.ipynb` por completo. Isso irá treinar o Autoencoder MLP e gerar o modelo `.tflite` quantizado (exportado automaticamente para a pasta `model/`).

### Parte 2: Deploy na Raspberry Pi Pico 2 W (Usando VS Code)
1. Instale o firmware do **MicroPython** na sua Raspberry Pi Pico 2 W.
2. No VS Code, instale a extensão **MicroPico**.
3. Conecte a placa via USB. A barra inferior do VS Code mostrará "Pico Connected".
4. Abra o script de execução em `pico/main.py`.
5. Clique com o botão direito sobre o código e selecione **Run current file on Pico** (ou faça o upload do arquivo usando a opção Upload to Pico).
6. Acompanhe a latência (medida na casa dos milissegundos) e o Erro Quadrático Médio (MSE) sendo impressos em tempo real no terminal serial do VS Code.


## 📄 Links e Documentação do Projeto

* 📑 **Artigo Científico (Formato SBC):** [Acessar Artigo no Overleaf](https://www.overleaf.com/read/zcdjpmtqhhxw#cd7944)
* 📊 **Apresentação de Slides:** [Acessar Apresentação no Google Drive](https://drive.google.com/drive/folders/1ZbwKHJKgELofY2mvI3emEaRZVV5U0Mlc?usp=sharing)
* ▶️ **Notebook:** [Acessar Arquivo 'energy_anomaly_detection.ipynb' no Google Colab](https://colab.research.google.com/drive/1jafGcsB_AHTifLLR1fu9cIKeldUXCd-9?usp=sharing)
