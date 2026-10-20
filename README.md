# Detecção de fraudes em rastreadores veiculares

Código-fonte do Trabalho de Conclusão de Curso **"Desenvolvimento e simulação de um sistema de detecção de fraudes em rastreadores veiculares"**, Bacharelado em Engenharia Eletrônica, CEFET/RJ.

**Autor:** Gustavo Luiz Ribeiro da Cruz

## Sobre o projeto

Rastreadores veiculares registram posição, velocidade, tensão de alimentação, sinal GSM/GPS e eventos (Reason Codes). Fraudes como o uso de bloqueadores de sinal (jammer) ou a desconexão da alimentação do equipamento deixam padrões nesses registros.

O sistema recebe a planilha de auditoria de **um rastreador** e responde **FRAUDE** ou **NÃO FRAUDE**, com o risco e a confiança de cada um dos três modelos comparados:

- Regressão Logística
- Rede Neural (MLP)
- XGBoost

## Como avaliar o projeto

Tudo roda no **Google Colab**, sem instalar nada no computador. É preciso apenas uma conta Google.

### Passo 1: acesso aos dados

A base de dados não está no GitHub por causa do volume e da confidencialidade. Ela fica no Google Drive:

**[Pasta da base de dados](https://drive.google.com/drive/folders/1vWRL0WtCI7Sl2JTjXNsoze-_mW5xlLAp?usp=drive_link)**

1. Abra o link com a conta Google que recebeu o acesso.
2. Clique com o botão direito na pasta e escolha **Organizar → Adicionar atalho ao Drive**, colocando o atalho em **Meu Drive**.
3. Confira se a estrutura no seu Drive ficou assim:

```
Meu Drive/
├── dados_brutos/                    planilhas de auditoria originais (*_auditoria.csv)
└── dataset_tratado/                 planilhas tratadas (*_treated.csv)
    └── resultados/
        ├── modelos_finais.pkl       modelos já treinados
        └── figuras/                 figuras da monografia
```

Os notebooks procuram exatamente esses nomes de pasta.

### Passo 2: abrir o notebook

Abra [deteccao_fraude_rastreadores.ipynb](deteccao_fraude_rastreadores.ipynb) e clique em **Abrir no Colab**, ou envie o arquivo pelo menu do Colab em **Arquivo → Fazer upload de notebook**.

### Passo 3: executar

1. No Colab, clique em **Ambiente de execução → Executar tudo**.
2. Autorize o acesso ao Google Drive quando o Colab pedir.

O notebook é autossuficiente: todo o código está dentro dele, não é preciso clonar repositório nem instalar bibliotecas manualmente.

### O que cada seção faz

| Seção | Conteúdo | Tempo aproximado |
|---|---|---|
| 0 | preparação do ambiente e pastas do Drive | segundos |
| 1 | tratamento das planilhas de auditoria | 2 a 5 min |
| 2 | base de janelas de 60 minutos por rastreador | 10 a 30 min |
| 3 | matriz de variáveis e definição dos modelos | segundos |
| 4 | Experimento A: generalização temporal | minutos |
| 5 | Experimento B: rastreadores nunca vistos | 20 a 40 min |
| 6 | decisão final por rastreador | segundos |
| 7 | figuras para a monografia | 1 a 2 min |
| 8 | treinamento dos modelos finais | minutos |
| 9 | tabela de métricas | segundos |
| 10 | classificação de um arquivo de rastreador | segundos |
| 11 | ranking de uma pasta | depende da quantidade de arquivos |

### Avaliação rápida

Para verificar apenas o resultado do classificador, rode as seções **0, 8 e 10**. A seção 8 carrega ou treina os modelos finais e a seção 10 analisa um arquivo e mostra o veredito dos três modelos.

O arquivo de exemplo usado é o **69251992613cd7d158638cb9_auditoria.csv**, um caso de fraude confirmada. Se ele não estiver no seu Drive, use a célula de envio do computador na própria seção 10.

## Metodologia

1. **Tratamento dos dados:** datas e durações são convertidas para tipos temporais antes do one-hot-encoding, preservando data, hora e minuto. São corrigidas coordenadas com separador de milhar, acentuação corrompida e linhas com ";" dentro do endereço.
2. **Unidade de análise:** janelas de 60 minutos de um único rastreador. As variáveis comparam o rastreador com ele mesmo (tensão relativa à própria mediana, área habitual de circulação, duração das condições de sinal).
3. **Rótulo de fraude:** jammer com pelo menos 30 minutos e/ou violação de alimentação (energia externa cortada enquanto o veículo se move com a ignição desligada). Um arquivo de fraudes confirmadas em `resultados/fraudes_confirmadas.csv`, com as colunas `rastreador;inicio;fim`, substitui o rótulo automático.
4. **Reason Codes:** eventos de fabricantes diferentes (CalAmp, Concox, MXT, Newtec, Sertec, Suntech e Queclink) agrupados em famílias: falha de energia, antena GPS removida, bateria de backup removida, desligamento, entre outras.
5. **Avaliação sem vazamento de dados:** imputação e padronização dentro de um Pipeline ajustado somente no conjunto de treino. Dois protocolos:
   - **temporal:** treino no período inicial e teste no período final;
   - **entre rastreadores:** validação cruzada por grupos (StratifiedGroupKFold), em que nenhum rastreador do teste aparece no treino.
6. **Limiar de decisão:** escolhido na validação para atingir precisão mínima de 50%.
7. **Decisão final:** um arquivo só é classificado como fraude se algum episódio reunir pelo menos dois tipos de indício ao mesmo tempo e o risco combinado atingir 50%. Indícios isolados não geram alarme, o que reduz os falsos positivos.

## Resultados gerados no Drive

Pasta `dataset_tratado/resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `tabela_metricas.csv` | precisão, recall, F1, PR-AUC e ROC-AUC dos três modelos |
| `experimento_A.csv` | métricas do teste temporal |
| `experimento_B_folds.csv` | métricas de cada fold da validação por rastreador |
| `decisao_por_rastreador.csv` | resultado da decisão final por rastreador |
| `composicao_folds.csv` | distribuição das fraudes entre os folds |
| `ranking_rastreadores.csv` | risco médio por rastreador de uma pasta |
| `modelos_finais.pkl` | modelos treinados, variáveis e regra de decisão |
| `figuras/` | figuras da monografia, em PNG de 300 dpi |

As figuras têm o prefixo do modelo no nome: `rl_` (Regressão Logística), `mlp_` (Rede Neural) e `xgb_` (XGBoost). São matrizes de confusão, curvas ROC, curvas precisão-recall, importância das variáveis e gráficos comparativos entre os três modelos.

## Dados de exemplo no repositório

A pasta `exemplos/` traz duas planilhas de auditoria reais, usadas para validar o sistema:

| Arquivo | O que é |
|---|---|
| `exemplos/69251992613cd7d158638cb9_auditoria.csv` | caso de **fraude confirmada**: oito episódios de violação de alimentação, com o veículo andando a mais de 80 km/h com a ignição desligada e a energia externa cortada |
| `exemplos/65c4c75f8cc6500f47d36bc1_auditoria.csv` | rastreador **sem fraude**: apenas interferências passageiras de sinal, nenhuma com mais de 5 minutos |

São dados operacionais já anonimizados, incluídos aqui para que a banca possa executar o classificador sem depender do Drive.

## Estrutura do repositório

```
├── deteccao_fraude_rastreadores.ipynb   notebook com todo o projeto
├── exemplos/                            planilhas de exemplo
├── README.md
├── requirements.txt                     bibliotecas do notebook
└── .gitignore
```

## Problemas comuns

| Mensagem | Solução |
|---|---|
| `FileNotFoundError` numa pasta do Drive | o atalho do Passo 1 não está em Meu Drive, ou os nomes das pastas foram alterados |
| `NameError` citando um nome que não é de função | rode a seção 0 antes das demais; ela define as constantes e importações usadas em todo o notebook |
| `KeyError` numa coluna | rode a seção que produz aquela tabela antes da seção que a usa |
| sessão encerrada por falta de memória | em **Ambiente de execução → Desconectar e excluir**, reabra e execute de novo; a base em cache evita reprocessar os arquivos |
| demora na seção 5 | reduza `N_REPETICOES` para 1 na seção 0, ou use `FOLDS_COMPLETOS = False` na seção 5 |
