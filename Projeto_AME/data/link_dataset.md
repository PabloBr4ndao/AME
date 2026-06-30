# Base de Dados: UK-DALE (UK Domestic Appliance-Level Electricity)

Este projeto baseia-se nas estatísticas empíricas de consumo da **Casa 1** do dataset público UK-DALE.

Devido ao tamanho massivo do dataset original (arquivos de alta e baixa frequência totalizando dezenas de Gigabytes), os dados brutos não estão hospedados neste repositório.

## Como os dados são gerados no projeto:
O arquivo `notebook/energy_anomaly_detection.ipynb` contém um gerador estatístico que sintetiza uma série temporal altamente realista, espelhando os limites e o comportamento do UK-DALE (consumo basal de 50W-150W e picos normais de aparelhos). A injeção de anomalias ocorre estritamente na base de testes para validação não-supervisionada do Autoencoder.

## Link Oficial para Download (Dataset Bruto):
Caso deseje acessar os dados originais completos utilizados como referência para este trabalho, acesse a publicação científica oficial:
* **Autores:** Kelly, J. and Knottenbelt, W. (2015). 
* **Publicação:** *Scientific Data*
* **Acesso:** [https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/UK-DALE](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/UK-DALE)