# Relatório de Prática: Sistema de Controle de Acesso Inteligente com RFID e Deep Learning

**Disciplina:** Aprendizado de Máquina Embarcado  
**Dupla:** Pablo Brandão Passos (539730) e Ronald Matheus Viana Lopes (541492)  
**Plataforma:** Raspberry Pi Pico 2W (RP2350)  

---

## 1. Visão Geral do Projeto

Este projeto consiste no desenvolvimento de um sistema de controle de acesso inteligente baseado em TinyML. Ao contrário de catracas tradicionais que apenas validam um identificador único (UID) em uma lista branca, nosso sistema emprega uma rede neural profunda (LSTM) para realizar o reconhecimento comportamental do usuário[cite: 2].

O modelo aprende o padrão de uso temporal de cada cartão (horário de acesso, dia da semana, intervalo entre acessos e frequência)[cite: 3]. Na ocorrência de anomalias — como uma tentativa de acesso fora do horário comercial habitual, um ataque de força bruta ou suspeita de clonagem — o sistema rejeita o acesso mesmo que o cartão seja válido no banco de dados[cite: 2, 3]. Todo o monitoramento do sistema e a interface de resposta foram adaptados para execução via terminal (UART) do computador.

---

## 2. Hardware e Fiação (Wiring)

Para esta prática, o hardware foi otimizado e focado nos componentes essenciais de coleta e alerta, removendo displays e atuadores mecânicos. Utilizamos o módulo leitor RFID RC522 comunicando-se via protocolo SPI e um Buzzer Ativo[cite: 2, 7].

### Tabela de Conexões

**Módulo RFID RC522 (Alimentação estrita de 3.3V)**
| Pino RC522 | Pino Pico 2W | Observação |
| :--- | :--- | :--- |
| SDA (CS) | GP5 | Chip Select do SPI0 |
| SCK | GP2 | Clock do SPI0 |
| MOSI | GP3 | Master Out Slave In do SPI0 |
| MISO | GP4 | Master In Slave Out do SPI0 |
| RST | GP0 | Reset |
| 3.3V | 3V3(OUT) | **NÃO ligar no 5V (VBUS)** |
| GND | GND | Trilho de terra |
| IRQ | *Desconectado* | A biblioteca funciona via *polling* |

**Buzzer Ativo**
| Pino Buzzer | Pino Pico 2W | Observação |
| :--- | :--- | :--- |
| Sinal (+) | GP18 | Acionado via PWM para bipes |
| GND (-) | GND | Trilho de terra |

---

## 3. Pipeline de Inteligência Artificial

A arquitetura de Machine Learning foi desenhada para operar sob severas restrições de memória[cite: 2]. O fluxo segue as seguintes etapas:

1. **Extração de Features:** A entrada da rede é uma janela deslizante contendo dados normalizados dos últimos 8 acessos: hora, dia, delta temporal e hash do UID[cite: 3, 8].
2. **Rede Neural LSTM:** Composta por duas camadas LSTM de 16 unidades seguidas por camadas Densas (ReLU e Softmax), a rede capta a dependência temporal dos acessos para classificar o evento em: NORMAL, SUSPEITO ou BLOQUEADO[cite: 2, 3, 8].
3. **Otimização (Poda e Quantização):** O treinamento ocorre no computador. Após o treino, aplicamos a técnica de poda (*pruning* por magnitude de 0.05), zerando os pesos de menor impacto[cite: 3, 8]. Em seguida, os parâmetros em *float32* são quantizados para *int8*[cite: 3, 8].
4. **Deploy:** Essas técnicas reduzem drasticamente o tamanho do modelo, permitindo embarcar toda a inteligência em um script minúsculo (`model.py`), que executa as inferências localmente na placa com baixa latência e consumo mínimo de RAM[cite: 2, 8].

---

## 4. Estrutura do Código e Funcionamento

O firmware principal (`main.py`) foi estruturado da seguinte forma:

* **Modo de Cadastro Automático:** Ao ligar a Pico, o sistema aguarda 10 segundos para o cadastro de novas tags RFID. Um bipe inicial é emitido e as informações são exibidas na UART. Os cartões são salvos em um arquivo local `rfid_db.json`.
* **Modo de Vigia:** Passado o tempo de cadastro, o leitor entra em modo de escuta.
* **Sistema de Alertas:** 
    * Se o MSE for baixo (acesso padrão), o sistema exibe "LIBERADO" no console e emite um bipe curto.
    * Se o MSE for mediano (acesso suspeito) ou alto (anomalia/bloqueio), o sistema emite no console a rejeição e dispara uma sequência de bipes de alerta pelo Buzzer.

---

## 5. Como Executar (Passo a Passo)

1. **Preparação do Modelo:**
   * Abra o terminal do computador na raiz do projeto.
   * Execute o script de treinamento: `python train_and_export.py`[cite: 8].
   * Verifique se o arquivo `model.py` foi gerado com sucesso[cite: 8].

2. **Upload para a Placa:**
   * Conecte a Raspberry Pi Pico 2W via USB.
   * Usando o VS Code com a extensão MicroPico, conecte-se à placa.
   * Faça o upload dos seguintes arquivos para o sistema de arquivos da Pico:
     - `main.py`
     - `rfid_rc522.py`
     - `model.py`

3. **Início da Operação:**
   * Inicie o arquivo `main.py` via MicroPico (`Run current file`).
   * Acompanhe o console UART aberto no rodapé do VS Code.
   * Aproxime o cartão na antena durante os primeiros 10 segundos para cadastrá-lo.
   * Passe o cartão novamente para testar a predição da Rede Neural.