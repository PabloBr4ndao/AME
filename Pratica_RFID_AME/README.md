# 🔐 Relatório Técnico: Controle de Acesso Inteligente com RFID e Deep Learning (TinyML)

**Disciplina:** Aprendizado de Máquina Embarcado  
**Dupla:** Pablo Brandão Passos (539730) e Ronald Matheus Viana Lopes (541492)  
**Plataforma de Hardware:** Raspberry Pi Pico 2W (Microcontrolador RP2350)  
**Linguagem de Programação:** MicroPython e Python 3  

---

## 1. Visão Geral e Proposta do Projeto

Este projeto consiste na implementação de um sistema de controle de acesso de alta segurança embarcado em uma Raspberry Pi Pico 2W. Diferente de catracas tradicionais que utilizam apenas validação condicional simples (verificação de UID em banco de dados), este sistema aplica **Inteligência Artificial (TinyML)** para detecção de anomalias comportamentais.

O sistema coleta dados de acesso, extrai atributos (*features*) e os repassa a uma Rede Neural Recorrente (LSTM) embarcada, que classifica o evento de acesso temporal em três categorias: **NORMAL, ATÍPICO ou BLOQUEADO**. A interface com o usuário foi adaptada e otimizada, trocando displays e servos mecânicos por comunicação serial (UART) direta via computador e feedback sonoro através de um Buzzer ativo.

---

## 2. Diagrama de Hardware e Fiação (Modificado)

O setup original do roteiro foi modificado e otimizado para manter o foco exclusivo no microcontrolador, na coleta SPI do leitor RFID e no processamento da inferência, eliminando as dependências I2C e portas GPIO em excesso.

### 2.1 Leitor MFRC522 (Interface SPI0)
O módulo de rádio frequência de 13.56 MHz exige comunicação rápida e alimentação estritamente regulada em 3.3V. O pino `IRQ` (Interrupt Request) foi intencionalmente deixado desconectado, uma vez que a leitura dos cartões foi implementada via varredura (*fast polling*).

| Pino do RC522 | Pino físico da Pico | GPIO | Função |
| :--- | :--- | :--- | :--- |
| **SDA (CS)** | Pino 7 | `GP5` | Chip Select / Slave Select |
| **SCK** | Pino 4 | `GP2` | Clock SPI |
| **MOSI** | Pino 5 | `GP3` | Master Out, Slave In |
| **MISO** | Pino 6 | `GP4` | Master In, Slave Out |
| **RST** | Pino 1 | `GP0` | Reset do módulo |
| **3.3V** | Pino 36 | `3V3(OUT)` | Alimentação do módulo |
| **GND** | Pino 38 | `GND` | Terra Comum |
| **IRQ** | - | - | *Não utilizado* |

### 2.2 Buzzer Ativo (Feedback Sonoro)
| Pino do Buzzer | Pino físico da Pico | GPIO | Função |
| :--- | :--- | :--- | :--- |
| **Sinal (+)** | Pino 24 | `GP18` | Controle de frequência via PWM |
| **GND (-)** | Pino 23 | `GND` | Terra Comum |

---

## 3. Pipeline de Inteligência Artificial e Deep Learning

### 3.1 Extração de Características (*Features*)
A entrada da rede neural não é o identificador bruto do cartão, mas sim o seu contexto temporal. Mantemos uma janela deslizante contendo os últimos **8 acessos** (`SEQ_LEN = 8`) de um usuário. Cada acesso possui 4 variáveis, normalizadas entre 0 e 1:
1. `hora_do_dia`: Posição atual da hora (dividido por 24).
2. `dia_da_semana`: Dia entre 0 e 6 (dividido por 7).
3. `delta_t`: Diferença de tempo entre o último acesso em segundos.
4. `uid_hash`: Checksum matemático do cartão RFID lido.

### 3.2 Arquitetura da Rede Neural
Utilizamos um modelo voltado a séries temporais desenvolvido puramente sobre matrizes (NumPy), sem o uso de frameworks pesados, para garantir a exportação eficiente para C/MicroPython.
* **Camada 1:** LSTM (16 unidades) - Capta o contexto temporal e a relação de dependência a longo prazo.
* **Camada 2:** LSTM (16 unidades) - Recebe o estado interno oculto da primeira camada.
* **Camada 3:** Densa com 8 Neurônios (Ativação ReLU).
* **Camada Saída:** Densa com 3 Neurônios (Ativação Softmax).

---

## 4. Treinamento, Resultados Obtidos e Compressão do Modelo

O treinamento do modelo ocorreu no computador, gerando logs reais das métricas e consolidando o arquivo final de pesos (`model.py`). Abaixo, reportamos os resultados quantitativos obtidos durante o processo.

### 4.1 Métricas de Treinamento
Foi gerado um dataset sintético com 600 instâncias de treino e 150 de validação, mapeando ataques de clonagem (rajadas), acessos noturnos incomuns e acessos em horário comercial.
A evolução em 2.000 épocas (SGD com *Learning Rate* de 0.005) retornou:
* **Acurácia (Treino):** 78.3%
* **Acurácia Final (Validação):** **74.7%**

### 4.2 Otimização I: Poda por Magnitude (*Pruning*)
Microcontroladores possuem poucos kilobytes de RAM estática livre. Para otimizar, aplicamos uma técnica de poda desativando todas as sinapses cujo valor de peso fosse inferior a `0.05` ($|w| < 0.05$).
* **Pesos Zerados:** 617 de 3619 parâmetros (17.0% da rede removida).
* **Acurácia Pós-Poda:** **74.0%** (queda negligenciável de 0.7%, confirmando a eficácia da técnica em ignorar pesos mortos).

### 4.3 Otimização II: Quantização (*Float32 para Int8*)
Após a poda, a rede realizava cálculos em ponto flutuante de 32-bits (Float32). O modelo passou por conversão e fator de escala (*Scaling Factor* = `max(|W|) / 127`) para uso de números inteiros limitados entre -128 e 127.
* **Redução na Memória:** De **14.476 Bytes** (14.4 KB) para apenas **3.619 Bytes** (3.6 KB).
* **Ganho:** Uma rede neuronal LSTM complexa capaz de predizer séries temporais comprimida em uma matriz de inteiros **4 vezes menor**, viabilizando o deploy na memória SRAM limitadíssima da Pico 2W.

---

## 5. Implementação Embarcada (O Firmware)

A lógica embarcada no MicroPython opera em Máquina de Estados Finita:
1. **Modo Cadastro (Boot):** Ao iniciar, o script `main.py` roda por 10 segundos varrendo tags RFID desconhecidas e salvando-as em uma memória persistente (`rfid_db.json`).
2. **Modo Vigia (Polling Contínuo):** Após os 10 segundos, o laço infinito é ativado. Se um cartão é lido e não está no DB, o buzzer dispara alerta de rejeição e a UART informa "DESCONHECIDO".
3. **Módulo de Inferência:** Se o cartão existe, o tempo atual é injetado no vetor de inferência e a função `classify()` do `model.py` resolve o estado atual daquele cartão baseado no histórico salvo na RAM da placa.
4. **Alertas Sonoros:**
   * `NORMAL`: Bipe curto único.
   * `ATIPICO`: Bipe grave (800 Hz) e aviso na Serial.
   * `BLOQUEADO`: Três bipes agudos sequenciais indicando anomalia/rajada de acessos.

---

## 6. Procedimento de Reprodução da Prática

### Ambiente e Dependências:
* Extensão **MicroPico** instalada no Visual Studio Code (Windows).
* Firmware oficial `MicroPython` instalado na placa Raspberry Pi Pico 2.
* Python 3 instalado no computador (biblioteca NumPy).

### Como Rodar:
1. Execute `python train_and_export.py` no terminal do seu computador para gerar o dataset, treinar o modelo, realizar as podas e compilar o arquivo com os pesos gerados (`model.py`).
2. Conecte a Pico 2W, configure o projeto pela paleta do VS Code (`Ctrl+Shift+P -> MicroPico: Connect`).
3. Envie exatamente os seguintes três arquivos para o diretório raiz da placa utilizando a função *Upload file to Pico*:
   - `model.py` (Arquitetura compactada pelo PC).
   - `rfid_rc522.py` (Driver SPI do Módulo RFID).
   - `main.py` (Script de orquestração do sistema).
4. Abra o `main.py` e execute o código na placa (*Run current file*). Cadastre cartões durante os bipes iniciais e em seguida realize as simulações aproximando o cartão do leitor e acompanhando a impressão das probabilidades no terminal.