# Air Quality Fault Detection Using Machine Learning

# Fault Detection in Low-Cost Air Quality Sensors using Machine Learning

## Detecção de Falhas em Sensores de Qualidade do Ar utilizando Aprendizado de Máquina

---

## Sobre o projeto

Este repositório contém todo o código-fonte desenvolvido durante a dissertação de Mestrado em Ciência da Computação da Universidade Estadual do Oeste do Paraná (UNIOESTE).

A pesquisa propõe uma metodologia baseada em Aprendizado de Máquina para detecção automática de falhas em sensores de baixo custo utilizados no monitoramento da qualidade do ar.

Além do código-fonte, este repositório disponibiliza todos os scripts utilizados durante o desenvolvimento da pesquisa, incluindo a preparação dos dados, simulação das falhas, treinamento dos modelos de aprendizado de máquina, geração automática das características e produção dos resultados experimentais apresentados na dissertação.

Este repositório constitui o **material suplementar oficial da dissertação**, permitindo a reprodução dos experimentos descritos no trabalho.

---

# Informações acadêmicas

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

# Objetivo

Desenvolver uma metodologia capaz de detectar automaticamente falhas em sensores ambientais de baixo custo utilizando técnicas de Aprendizado de Máquina, aumentando a confiabilidade dos dados coletados por redes de monitoramento da qualidade do ar.

---

# Variáveis ambientais analisadas

- PM2.5
- PM10
- Temperatura
- Umidade

---

# Falhas simuladas

Os experimentos contemplam a detecção automática das seguintes categorias de falhas:

- Oscilação
- Queda
- Lacuna

As falhas foram inseridas automaticamente em séries temporais ambientais para avaliar a capacidade dos modelos de aprendizado de máquina em reconhecer eventos anômalos.

---

# Modelos de Aprendizado de Máquina

Foram avaliados os seguintes algoritmos:

- Random Forest
- XGBoost
- CatBoost
- Multi-Layer Perceptron (MLP)

---

# Fluxo geral da metodologia

```text
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

# Estrutura do repositório

```text
air-quality-fault-detection-ml/
│
├── dataError_IC-master/
│   Código principal da dissertação
│
├── graficos_artigo_mp25_1/
│   Figuras utilizadas na dissertação
│
├── raw_data_DUSTAI/
│   Base de dados utilizada nos experimentos
│
├── main e ajustes originais/
│   Versões iniciais do desenvolvimento
│
├── LICENSE
│
└── README.md
```

---

# Código principal da dissertação

O desenvolvimento principal desta pesquisa encontra-se na pasta:

```text
dataError_IC-master/
```

Nessa pasta estão implementados:

- simulação das falhas;
- processamento das séries temporais;
- geração automática das características;
- treinamento dos modelos;
- avaliação dos classificadores;
- geração automática das tabelas;
- geração automática dos gráficos.

---

# Código dos experimentos multivariáveis

O módulo responsável pelos experimentos realizados com as quatro variáveis ambientais encontra-se em:

```text
dataError_IC-master/
└── PROJETO MULTIVARIÁVEL/
```

Esse módulo reúne:

- experimentos para PM2.5;
- experimentos para PM10;
- experimentos para Temperatura;
- experimentos para Umidade;
- cinco rodadas experimentais;
- relatórios completos;
- resultados gerados automaticamente;
- tabelas utilizadas na dissertação.

---

# Resultados disponibilizados

O repositório disponibiliza os resultados completos produzidos durante os experimentos, incluindo:

- arquivos CSV;
- métricas de desempenho;
- matrizes de confusão;
- tabelas completas;
- figuras;
- relatórios.

Todos os resultados correspondem aos experimentos descritos na dissertação.

---

# Reprodutibilidade

Todos os experimentos apresentados na dissertação podem ser reproduzidos utilizando os códigos disponibilizados neste repositório.

Os scripts permitem reproduzir:

- simulação das falhas;
- geração automática das características;
- treinamento dos modelos;
- avaliação dos classificadores;
- geração das métricas;
- geração das tabelas;
- geração das figuras.

---

# Tecnologias utilizadas

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib
- PyCharm

---

# Como citar este trabalho

Caso este repositório seja utilizado em pesquisas acadêmicas, recomenda-se citar a dissertação correspondente.

**Rodrigues, Vaniely.**

*Detecção de Falhas em Sensores de Qualidade do Ar utilizando Aprendizado de Máquina.*

Programa de Pós-Graduação em Ciência da Computação.

Universidade Estadual do Oeste do Paraná (UNIOESTE).

2026.

---

# Contato

Vaniely Rodrigues

Programa de Pós-Graduação em Ciência da Computação

Universidade Estadual do Oeste do Paraná (UNIOESTE)

---

# Licença

Este projeto está disponível sob os termos da licença MIT.

Consulte o arquivo [LICENSE](LICENSE) para mais informações.
