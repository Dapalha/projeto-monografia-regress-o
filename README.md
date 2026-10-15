# Detecção de fraudes em rastreadores veiculares

Código do Trabalho de Conclusão de Curso *Desenvolvimento e simulação de um sistema de detecção de fraudes em rastreadores veiculares*, Engenharia Eletrônica, CEFET/RJ.

O sistema recebe a planilha de auditoria de um rastreador e indica se houve fraude, comparando três classificadores: **Regressão Logística**, **Rede Neural (MLP)** e **XGBoost**.

## Notebooks

| Notebook | Conteúdo | Colab |
|---|---|---|
| [01_treinamento_e_avaliacao](notebooks/01_treinamento_e_avaliacao.ipynb) | tratamento dos dados, montagem da base, experimentos, figuras e modelos finais | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dapalha/projeto-monografia-regressao/blob/main/notebooks/01_treinamento_e_avaliacao.ipynb) |
| [02_classificador](notebooks/02_classificador.ipynb) | classificação de um arquivo, ranking de uma pasta e interface web | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dapalha/projeto-monografia-regressao/blob/main/notebooks/02_classificador.ipynb) |

## Estrutura

```
projeto-monografia-regressao/
├── fraude/                  pacote com o código reutilizável
│   ├── tratamento.py        leitura e limpeza das planilhas de auditoria
│   ├── features.py          variáveis por registro, janelas e decisão combinada
│   ├── base.py              montagem da base e rótulos confirmados
│   ├── modelos.py           Regressão Logística, MLP e XGBoost
│   ├── avaliacao.py         métricas por janela e por rastreador
│   ├── figuras.py           figuras da monografia
│   ├── analise.py           classificação de um arquivo
│   └── interface.py         interface web (Gradio)
├── notebooks/
├── app.py                   interface web fora do Colab
└── requirements.txt
```

## Dados

A base não fica no repositório por causa do volume e da confidencialidade. Estrutura esperada no Google Drive:

```
Meu Drive/
├── dados_brutos/            planilhas *_auditoria.csv ou .xlsx
└── dataset_tratado/         *_treated.csv (gerados pelo notebook 01)
    └── resultados/
        ├── fraudes_confirmadas.csv   opcional: rastreador;inicio;fim
        ├── modelos_finais.pkl
        └── figuras/
```

Link da base: https://drive.google.com/drive/folders/1vWRL0WtCI7Sl2JTjXNsoze-_mW5xlLAp?usp=drive_link

## Metodologia

- **Tratamento:** datas e durações viram tipos temporais antes do one-hot-encoding; coordenadas, acentuação e linhas com ';' extra são corrigidas.
- **Unidade de análise:** janelas de 60 minutos de um único rastreador, com variáveis calculadas em relação ao próprio rastreador.
- **Rótulo:** jammer com duração de pelo menos 30 minutos e/ou violação de alimentação (energia externa cortada com o veículo em movimento e ignição desligada). Períodos confirmados, quando disponíveis, substituem a regra.
- **Reason Codes:** códigos de diferentes fabricantes (CalAmp, Concox, MXT, Newtec, Sertec, Suntech, Queclink) agrupados em famílias de evento.
- **Avaliação sem vazamento:** imputação e padronização dentro de um Pipeline ajustado só no treino; protocolo temporal e protocolo por grupos (StratifiedGroupKFold).
- **Limiar:** escolhido na validação para atingir precisão mínima de 50%.
- **Decisão:** um arquivo só é classificado como fraude se algum episódio reunir pelo menos dois tipos de indício e o risco combinado atingir 50%.

## Execução

1. Abra o notebook 01 no Colab e autorize o acesso ao Google Drive.
2. Na primeira execução, use TRATAR_DADOS_BRUTOS = True e RECONSTRUIR_BASE = True.
3. Execute todas as células. As figuras ficam em resultados/figuras e os modelos em resultados/modelos_finais.pkl.
4. Abra o notebook 02 para classificar arquivos ou abrir a interface.

Para a interface fora do Colab:

```bash
pip install -r requirements.txt
python app.py
```

O modelo salvo deve ser aberto com as mesmas versões de bibliotecas usadas no treinamento.
