# PROJETO-MESTRADO

# Fault Detection in Low-Cost Air Quality Sensors using Machine Learning

## Detecção de Falhas em Sensores de Qualidade do Ar utilizando Aprendizado de Máquina

---

## Sobre o projeto

Este repositório contém todo o código-fonte desenvolvido durante a dissertação de Mestrado em Ciência da Computação da Universidade Estadual do Oeste do Paraná (UNIOESTE).

O trabalho propõe uma metodologia baseada em Aprendizado de Máquina para detectar automaticamente falhas em sensores de baixo custo utilizados no monitoramento da qualidade do ar.

Além do código-fonte, este repositório disponibiliza os scripts utilizados na preparação dos dados, simulação das falhas, treinamento dos modelos de aprendizado de máquina e geração automática dos resultados experimentais apresentados na dissertação.

---

## Informações acadêmicas

**Título da dissertação**

Detecção de Falhas em Sensores de Qualidade do Ar utilizando Aprendizado de Máquina

**Programa**

Programa de Pós-Graduação em Ciência da Computação (PPGComp)

**Instituição**

Universidade Estadual do Oeste do Paraná (UNIOESTE)

**Autora**

Vaniely Rodrigues

**Orientador**

Prof. Dr. Roberto Sheffel

---

## Objetivo

O objetivo deste trabalho é desenvolver uma metodologia capaz de detectar automaticamente falhas em sensores ambientais de baixo custo utilizando técnicas de Aprendizado de Máquina, contribuindo para aumentar a confiabilidade dos dados coletados em redes de monitoramento da qualidade do ar.

---

## Variáveis ambientais analisadas

- PM2.5
- PM10
- Temperatura
- Umidade

---

## Falhas simuladas

Durante os experimentos foram simuladas três categorias de falhas em séries temporais ambientais:

- Oscilação
- Queda
- Lacuna

Cada falha foi inserida automaticamente em diferentes posições das séries temporais, permitindo avaliar a capacidade dos modelos em identificar eventos anômalos.

---

## Modelos de Aprendizado de Máquina

Foram avaliados os seguintes algoritmos:

- Random Forest
- XGBoost
- CatBoost
- Multi-Layer Perceptron (MLP)

---

## Fluxo geral da metodologia

```
Dados ambientais
        │
        ▼
Simulação das falhas
        │
        ▼
Pré-processamento
        │
        ▼
Extração de características
        │
        ▼
Treinamento dos modelos
        │
        ▼
Avaliação dos resultados
```

---

## Estrutura do repositório

```
PROJETO-MESTRADO
│
├── dataError_IC-master
│   Código principal do projeto
│
├── graficos_artigo_mp25_1
│   Figuras utilizadas na dissertação
│
├── raw_data_DUSTAI
│   Base de dados utilizada nos experimentos
│
├── main e ajustes originais
│   Versões iniciais do desenvolvimento
│
└── README.md
```

---

## Resultados disponibilizados

O repositório contém os resultados completos obtidos durante os experimentos realizados para as quatro variáveis ambientais, incluindo:

- cinco rodadas experimentais;
- métricas de desempenho;
- tabelas completas;
- arquivos CSV gerados automaticamente;
- figuras utilizadas na dissertação.

---

## Reprodutibilidade

Todos os experimentos descritos na dissertação podem ser reproduzidos a partir dos códigos disponibilizados neste repositório.

Os scripts permitem reproduzir:

- simulação das falhas;
- geração das características;
- treinamento dos modelos;
- validação dos modelos;
- geração automática das métricas;
- geração das tabelas utilizadas na dissertação.

---

## Tecnologias utilizadas

- Python 3
- Scikit-learn
- XGBoost
- CatBoost
- NumPy
- Pandas
- Matplotlib
- PyCharm

---

## Publicação

Este repositório foi desenvolvido como material suplementar da dissertação de Mestrado.

Caso este código seja utilizado em pesquisas acadêmicas, solicita-se a citação da dissertação correspondente.

---

## Contato

Vaniely Rodrigues

Programa de Pós-Graduação em Ciência da Computação

Universidade Estadual do Oeste do Paraná (UNIOESTE)

---

## Licença

Este projeto é disponibilizado exclusivamente para fins acadêmicos e científicos.
