# SENAI Miniprojeto MNIST

## Classificação de Dígitos Manuscritos com Machine Learning

Este projeto foi desenvolvido como miniprojeto de Machine Learning utilizando o conjunto de dados **MNIST (Modified National Institute of Standards and Technology)**.

O objetivo principal é desenvolver, comparar e avaliar diferentes modelos de classificação capazes de reconhecer dígitos manuscritos de **0 a 9**, explorando não apenas métricas tradicionais de desempenho, mas também aspectos relacionados à **generalização, dados fora da distribuição (OOD), confiança das previsões e inferência com imagens próprias**.

Foram avaliados três algoritmos:

- K-Nearest Neighbors (KNN);
- Random Forest;
- Multilayer Perceptron (MLP).

O projeto foi desenvolvido em **Python utilizando Jupyter Notebook** e versionado com **Git/GitHub**, adotando branches específicas para cada etapa do desenvolvimento.

---

# 1. Objetivos

Os principais objetivos do projeto são:

- compreender a estrutura do conjunto MNIST;
- realizar análise exploratória dos dados;
- preparar e normalizar as imagens;
- dividir corretamente os dados em treinamento, validação e teste;
- treinar diferentes algoritmos de classificação;
- realizar seleção de hiperparâmetros;
- comparar os modelos utilizando diferentes métricas;
- selecionar o melhor classificador;
- analisar matrizes de confusão e erros de classificação;
- avaliar o comportamento diante de classes ocultas;
- discutir o conceito de overconfidence;
- testar o modelo com imagens externas próprias;
- avaliar a influência do método de aquisição e da espessura do traço;
- analisar limitações de generalização do modelo.

---

# 2. Estrutura do Projeto

O projeto foi organizado separando as imagens utilizadas nos testes,
as imagens processadas, o modelo treinado, o notebook principal e as
figuras geradas durante as etapas de avaliação.

```text
SENAI-Miniprojeto-MNIST/
│
├── images/
│   │
│   ├── raw/
│   │   ├── digito_5camf.jpg
│   │   ├── digito_5camg.jpg
│   │   ├── digito_5camm.jpg
│   │   ├── digito_5cf.jpg
│   │   ├── digito_5cgj.jpg
│   │   ├── digito_5M.jpg
│   │   ├── digito_5mc.jpg
│   │   ├── digito_5P.jpg
│   │   ├── digito_5PaintBf.jpg
│   │   ├── digito_5PaintBg.jpg
│   │   ├── digito_5PaintBm.jpg
│   │   ├── digito_5PaintF.jpg
│   │   ├── digito_5PaintG.jpg
│   │   └── digito_5PaintM.jpg
│   │
│   └── processed/
│       ├── comparacao_imagens_processadas.png
│       ├── predicao_celular_fino.png
│       ├── predicao_celular_grosso.png
│       ├── predicao_celular_medio.png
│       ├── predicao_paint_fino.png
│       ├── predicao_paint_grosso.png
│       ├── predicao_paint_medio.png
│       ├── predicao_scanner_fino.png
│       ├── predicao_scanner_grosso.png
│       └── predicao_scanner_medio.png
│
├── models/
│   └── mlp_final.joblib
│
├── notebook/
│   └── mnist.ipynb
│
├── reports/
│   └── figures/
│       ├── fase_5_3/
│       ├── comparacao_metricas.png
│       ├── comparacao_tempo_predicao.png
│       ├── comparacao_tempo_treino.png
│       ├── matriz_confusao_knn.png
│       ├── matriz_confusao_mlp.png
│       └── matriz_confusao_random_forest.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Descrição dos diretórios

- **`images/raw/`** — contém as imagens próprias originais utilizadas
  nos experimentos de inferência, incluindo imagens provenientes de
  celular, scanner e Paint.

- **`images/processed/`** — contém as imagens geradas após o
  pré-processamento e as visualizações das previsões realizadas pelo
  modelo. Também inclui a comparação das imagens processadas e os
  resultados individuais de celular, scanner, Paint e imagens
  manuscritas utilizadas nos testes.

- **`models/`** — contém o modelo final selecionado após a comparação
  dos classificadores. O arquivo `mlp_final.joblib` armazena o pipeline
  final da rede neural MLP.

- **`notebook/`** — contém o notebook principal `mnist.ipynb`, onde são
  executadas e documentadas todas as etapas do projeto.

- **`reports/figures/`** — contém as figuras utilizadas na análise dos
  resultados, incluindo matrizes de confusão e gráficos de comparação
  de métricas e tempos computacionais.

- **`images/processed/`** — contém figuras específicas dos testes
  realizados na Fase 5.3 com imagens próprias.

- **`.gitignore`** — define arquivos e diretórios que não devem ser
  versionados.

- **`README.md`** — contém a documentação geral do projeto.

- **`requirements.txt`** — registra as dependências necessárias para
  execução do projeto.

---

# 3. Tecnologias Utilizadas

O projeto utiliza principalmente:

- Python;
- Jupyter Notebook;
- NumPy;
- Pandas;
- Matplotlib;
- Scikit-learn;
- Pillow (PIL);
- Git;
- GitHub.

---

# 4. Dataset MNIST

O MNIST é um conjunto de dados amplamente utilizado para problemas de classificação de imagens.

O conjunto contém **70.000 imagens de dígitos manuscritos**, representando as classes de **0 a 9**.

Cada imagem possui dimensão:

```text
28 × 28 pixels
```

Para utilização pelos algoritmos de Machine Learning, cada imagem é representada vetorialmente:

```text
28 × 28 = 784 atributos
```

Cada atributo corresponde à intensidade de um pixel.

Originalmente, os pixels apresentam valores entre:

```text
0 e 255
```

onde valores menores representam pixels mais escuros e valores maiores representam maior intensidade.

---

# 5. Fases do Projeto

## Fase 1 — Análise Exploratória dos Dados (EDA)

A primeira etapa teve como objetivo compreender a estrutura e as características do MNIST.

Foram analisados:

- número total de amostras;
- quantidade de atributos;
- dimensões das imagens;
- distribuição das classes;
- escala de intensidade dos pixels;
- exemplos de imagens;
- representação vetorial das imagens.

Cada imagem bidimensional de `28 × 28` pixels é convertida em um vetor contendo `784 features`.

A análise exploratória permitiu verificar a distribuição das classes e compreender como os dados seriam utilizados pelos classificadores.

---

# 6. Fase 2 — Pré-processamento

O pré-processamento teve como objetivo preparar os dados para o treinamento dos modelos.

Os valores dos pixels foram normalizados do intervalo:

```text
[0, 255]
```

para:

```text
[0.0, 1.0]
```

utilizando:

```python
def normalizar_pixels(X):
    return (
        X.astype(np.float32)
        / 255.0
    )
```

A normalização ajuda os modelos, especialmente redes neurais, a trabalhar com valores em uma escala padronizada.

---

## Divisão dos dados

O conjunto foi separado em conjuntos distintos para desenvolvimento e avaliação.

Durante a seleção de hiperparâmetros:

```text
Treinamento: 49.000 imagens
Validação:     7.000 imagens
```

Após a escolha das melhores configurações, os dados de treinamento e validação foram reunidos:

```text
Treinamento final: 56.000 imagens
Teste final:       14.000 imagens
```

Foi utilizado:

```python
RANDOM_STATE = 42
```

para permitir maior reprodutibilidade dos experimentos.

O conjunto de teste foi mantido separado durante a seleção dos modelos, sendo utilizado somente na avaliação final.

---

# 7. Fase 3 — Modelagem

Foram avaliados três algoritmos de classificação.

## 7.1 K-Nearest Neighbors — KNN

O KNN classifica uma nova amostra com base nas classes das amostras mais próximas.

A melhor configuração encontrada foi:

```python
{
    "n_neighbors": 3,
    "weights": "distance"
}
```

Nesse caso, foram utilizados três vizinhos e os vizinhos mais próximos receberam maior peso na decisão.

---

## 7.2 Random Forest

Random Forest combina diversas árvores de decisão para realizar a classificação.

A melhor configuração encontrada foi:

```python
{
    "max_depth": None,
    "n_estimators": 200,
    "n_jobs": -1,
    "random_state": 42
}
```

Foram utilizadas 200 árvores, sem limitação explícita de profundidade.

---

## 7.3 Multilayer Perceptron — MLP

A MLP é uma rede neural artificial composta por camadas de neurônios.

A melhor configuração encontrada foi:

```python
{
    "activation": "relu",
    "alpha": 0.001,
    "early_stopping": True,
    "hidden_layer_sizes": (128, 64),
    "max_iter": 100,
    "random_state": 42,
    "solver": "adam"
}
```

A arquitetura utilizada foi:

```text
Entrada
  │
  ├── 784 pixels
  │
  ▼
Camada oculta
128 neurônios
  │
  ▼
Camada oculta
64 neurônios
  │
  ▼
Saída
10 classes
```

Foi utilizada a função de ativação **ReLU**, otimizador **Adam** e regularização L2 com:

```text
alpha = 0.001
```

Também foi utilizado:

```python
early_stopping=True
```

permitindo interromper o treinamento quando a validação interna deixasse de apresentar melhora suficiente.

O treinamento final da MLP foi encerrado após aproximadamente **43 iterações**, antes do limite máximo de 100.

---

# 8. Fase 4 — Avaliação dos Modelos

Os três modelos foram avaliados utilizando:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- tempo de treinamento;
- tempo de predição;
- matriz de confusão.

Os resultados no conjunto de teste foram:

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **MLP** | **98,01%** | **98,01%** | **98,01%** | **98,01%** |
| KNN | 97,32% | 97,34% | 97,32% | 97,32% |
| Random Forest | 96,78% | 96,78% | 96,78% | 96,78% |

A MLP apresentou o melhor desempenho geral.

---

## 8.1 Tempo de treinamento

Os tempos finais de treinamento foram aproximadamente:

| Modelo | Tempo |
|---|---:|
| KNN | 0,18 s |
| Random Forest | 31,37 s |
| MLP | 64,05 s |

O KNN apresentou treinamento extremamente rápido, enquanto a MLP apresentou o maior custo de treinamento.

---

## 8.2 Tempo de predição

Os tempos observados foram aproximadamente:

| Modelo | Tempo de predição |
|---|---:|
| **MLP** | **0,13 s** |
| Random Forest | 0,71 s |
| KNN | 25,03 s |

Embora o KNN tenha treinamento rápido, sua inferência foi significativamente mais lenta.

A MLP apresentou a melhor combinação entre **desempenho preditivo e velocidade de inferência**.

---

# 9. Matrizes de Confusão

Foram construídas matrizes de confusão para os três classificadores.

As matrizes mostraram forte concentração de valores na diagonal principal, indicando elevado número de classificações corretas.

Também foram identificadas confusões entre dígitos visualmente semelhantes.

As figuras geradas incluem:

```text
matriz_confusao_knn.png
matriz_confusao_random_forest.png
matriz_confusao_mlp.png
```

A análise das matrizes complementa as métricas globais, permitindo identificar quais classes apresentam maior dificuldade de classificação.

---

# 10. Modelo Final

Com base nos resultados obtidos, a **MLP foi selecionada como modelo final**.

Principais motivos:

- maior Accuracy;
- maior Precision;
- maior Recall;
- maior F1-score;
- excelente velocidade de inferência;
- boa capacidade de generalização no conjunto de teste MNIST.

O modelo final foi armazenado no diretório:

```text
models/
```

O pipeline contém tanto a etapa de normalização quanto o classificador MLP.

Isso é importante porque garante que novas entradas sejam submetidas ao mesmo processo de normalização utilizado durante o treinamento.

---

# 11. Fase 5 — Testes Avançados

Além da avaliação tradicional, foram realizados experimentos adicionais para investigar o comportamento do classificador em situações diferentes das encontradas no teste convencional do MNIST.

---

# 12. Fase 5.2 — Classes Ocultas e OOD

Foi realizado um experimento envolvendo classes ocultas, com o objetivo de observar o comportamento do classificador diante de exemplos não apresentados da mesma forma durante o treinamento.

Foram utilizadas como classes ocultas os dígitos:

```text
4 e 7
```

O experimento permitiu observar como o modelo força uma decisão entre as classes disponíveis mesmo quando recebe entradas provenientes das classes ocultadas.

A confiança observada apresentou:

```text
Confiança média:   0,9166176
Confiança máxima:  1,0000000
Confiança mínima:  0,32259402
```

A confiança média foi, portanto, de aproximadamente:

```text
91,66%
```

Esse resultado é importante porque demonstra que o modelo pode apresentar elevada confiança mesmo diante de situações em que sua previsão deve ser interpretada com cautela.

---

## Overconfidence

O conceito de **overconfidence**, ou falsa certeza, ocorre quando um modelo apresenta elevada confiança em uma previsão que pode estar incorreta.

Uma probabilidade elevada não representa necessariamente garantia de acerto.

Esse fenômeno é particularmente importante em aplicações reais de Inteligência Artificial, nas quais decisões automatizadas podem ser utilizadas em contextos sensíveis.

---

# 13. Fase 5.3 — Inferência com Imagens Próprias

A capacidade de generalização do modelo também foi avaliada utilizando imagens externas produzidas especificamente para o projeto.

O experimento foi ampliado para **nove imagens do dígito 5**.

Foram utilizados três métodos de aquisição:

1. fotografia com celular;
2. scanner a 300 dpi;
3. desenho digital no Paint.

Para cada método foram produzidas três espessuras de traço:

```text
Fino
Médio
Grosso
```

Dessa forma:

```text
                    FINO       MÉDIO       GROSSO

CELULAR               ✓           ✓           ✓
SCANNER               ✓           ✓           ✓
PAINT                 ✓           ✓           ✓
```

totalizando nove imagens.

---

# 14. Pré-processamento das Imagens Externas

As imagens externas passaram por um pipeline contendo:

1. conversão para escala de cinza;
2. inversão de cores quando necessária;
3. remoção de parte do ruído de fundo;
4. identificação do bounding box do dígito;
5. recorte da região de interesse;
6. redimensionamento mantendo a proporção;
7. centralização em um canvas 28 × 28;
8. preparação para entrada no modelo.

As fotografias de celular e imagens do scanner possuíam originalmente:

```text
fundo branco + traço escuro
```

e foram invertidas.

As imagens do Paint foram posteriormente produzidas com:

```text
fundo escuro + traço claro
```

não necessitando inversão.

As imagens externas foram mantidas na escala `[0,255]` ao final desse pré-processamento, pois o próprio pipeline do modelo final contém a função:

```python
def normalizar_pixels(X):
    return (
        X.astype(np.float32)
        / 255.0
    )
```

Dessa forma, foi evitada a **normalização duplicada**.

---

# 15. Resultados das Imagens Próprias

Os resultados definitivos foram:

| Método | Traço | Real | Previsto | Confiança | Resultado |
|---|---|---:|---:|---:|---|
| Celular | Fino | 5 | 2 | 46,69% | Incorreto |
| Celular | Médio | 5 | 6 | 99,66% | Incorreto |
| Celular | Grosso | 5 | 5 | 58,76% | Correto |
| Scanner | Fino | 5 | 4 | 29,44% | Incorreto |
| Scanner | Médio | 5 | 5 | 93,48% | Correto |
| Scanner | Grosso | 5 | 8 | 61,55% | Incorreto |
| Paint | Fino | 5 | 5 | 27,82% | Correto |
| Paint | Médio | 5 | 5 | 59,43% | Correto |
| Paint | Grosso | 5 | 5 | 94,59% | Correto |

Foram obtidos:

```text
5 acertos em 9 imagens
```

correspondendo a:

```text
55,56%
```

de acerto **neste pequeno conjunto experimental**.

Esse valor não deve ser interpretado como uma estimativa geral da acurácia do modelo para imagens externas, pois foram utilizadas apenas nove imagens e todas pertenciam à classe 5.

---

# 16. Comparação por Método de Aquisição

## Celular

```text
1 acerto / 3 imagens
```

Taxa experimental:

```text
33,33%
```

As imagens obtidas pelo celular apresentaram maior influência de iluminação, sombras, contraste e ruído de fundo.

---

## Scanner

```text
1 acerto / 3 imagens
```

Taxa experimental:

```text
33,33%
```

O scanner produziu imagens mais uniformes que o celular, porém diferenças de intensidade e espessura ainda afetaram a classificação.

---

## Paint

```text
3 acertos / 3 imagens
```

Taxa experimental:

```text
100%
```

As imagens produzidas digitalmente apresentaram comportamento mais consistente neste experimento.

Entretanto, devido ao pequeno número de imagens, esse resultado não permite concluir que imagens do Paint serão sempre mais fáceis de classificar.

---

# 17. Influência da Espessura do Traço

Nas imagens produzidas no Paint foi observado aumento progressivo da confiança:

```text
Fino    → 27,82%
Médio   → 59,43%
Grosso  → 94,59%
```

As três imagens foram corretamente classificadas como 5.

Para essas amostras específicas, o aumento da espessura do traço esteve associado ao aumento da confiança.

O mesmo padrão, entretanto, não ocorreu de maneira consistente nas imagens do celular e scanner.

Isso demonstra que a espessura não é o único fator relevante.

Também podem influenciar:

- iluminação;
- contraste;
- ruído;
- enquadramento;
- centralização;
- intensidade dos pixels;
- características do processo de aquisição.

---

# 18. Exemplo de Overconfidence nas Imagens Próprias

Um dos resultados mais importantes ocorreu com:

```text
Celular - Médio
```

O resultado foi:

```text
Dígito real:       5
Dígito previsto:   6
Confiança:        99,66%
```

Apesar da previsão estar incorreta, o modelo apresentou confiança extremamente elevada.

Esse resultado constitui um exemplo prático de:

```text
OVERCONFIDENCE
```

ou **falsa certeza**.

Ele demonstra que a confiança produzida pelo modelo não deve ser interpretada como garantia de que a previsão esteja correta.

---

# 19. Generalização

No conjunto de teste MNIST, a MLP apresentou aproximadamente:

```text
98,01% de acurácia
```

Entretanto, nas nove imagens externas do experimento foram obtidos:

```text
5 acertos em 9 imagens
```

Essa diferença demonstra um dos principais desafios de Machine Learning:

## Generalização para dados externos

Um modelo pode apresentar excelente desempenho quando os dados avaliados possuem características semelhantes às utilizadas durante seu desenvolvimento, mas apresentar comportamento diferente quando recebe dados provenientes de outras condições.

As imagens próprias apresentaram diferenças relacionadas a:

- iluminação;
- aquisição;
- resolução;
- ruído;
- espessura;
- contraste;
- intensidade;
- centralização.

Essas diferenças modificam a distribuição dos pixels apresentados à rede neural.

---

# 20. Principais Resultados

Os principais resultados do projeto foram:

### Melhor modelo

```text
MLP
```

### Accuracy no teste MNIST

```text
98,01%
```

### Melhor tempo de inferência

```text
MLP ≈ 0,13 s
```

### Imagens externas

```text
5 / 9 classificadas corretamente
```

### Exemplo de falsa certeza

```text
Real:       5
Previsto:   6
Confiança: 99,66%
```

---

# 21. Conclusão

O projeto permitiu desenvolver e comparar diferentes algoritmos de Machine Learning para classificação de dígitos manuscritos utilizando o MNIST.

Entre os modelos avaliados, a **Multilayer Perceptron (MLP)** apresentou o melhor desempenho geral, alcançando aproximadamente **98,01% de acurácia** no conjunto de teste e apresentando também excelentes valores de Precision, Recall e F1-score.

A comparação dos tempos computacionais mostrou diferenças importantes entre os algoritmos. O KNN apresentou treinamento extremamente rápido, porém inferência lenta. A Random Forest apresentou comportamento intermediário, enquanto a MLP exigiu maior tempo de treinamento, mas apresentou inferência rápida.

Dessa forma, a MLP foi selecionada como modelo final considerando conjuntamente **qualidade das previsões e desempenho durante a inferência**.

As matrizes de confusão demonstraram que, apesar do elevado desempenho, ainda existem erros entre dígitos com características visuais semelhantes.

Os experimentos com classes ocultas mostraram que o modelo pode apresentar elevada confiança mesmo diante de situações diferentes daquelas utilizadas durante seu treinamento.

Os testes com imagens próprias reforçaram essa conclusão. Embora todas as nove imagens representassem o mesmo dígito, diferenças no método de aquisição e na espessura dos traços produziram respostas significativamente diferentes.

Um dos resultados mais relevantes foi uma classificação incorreta com **99,66% de confiança**, demonstrando na prática o problema de **overconfidence**.

Portanto, o projeto evidencia que a avaliação de um sistema de Inteligência Artificial não deve se limitar apenas à acurácia obtida em um conjunto de teste.

Também é necessário analisar:

- capacidade de generalização;
- tipos de erros;
- comportamento diante de dados externos;
- consistência do pré-processamento;
- confiança das previsões;
- limitações do modelo.

Assim, além de permitir a implementação de classificadores para o MNIST, o projeto demonstrou conceitos fundamentais para o desenvolvimento responsável de sistemas de Machine Learning.

---

# 22. Possíveis Melhorias Futuras

Como evolução do projeto, poderiam ser investigadas:

- Redes Neurais Convolucionais (CNN);
- data augmentation;
- técnicas mais robustas de centralização;
- remoção automática de ruído;
- limiarização adaptativa;
- correção automática de iluminação;
- calibração das probabilidades;
- métodos específicos para detecção de dados OOD;
- aumento do número de imagens externas;
- testes externos envolvendo todos os dígitos de 0 a 9.

Uma CNN seria particularmente interessante, pois consegue explorar diretamente relações espaciais entre os pixels das imagens.

---

# 23. Versionamento com Git e GitHub

O projeto foi desenvolvido utilizando controle de versão com Git.

Foi utilizada a branch:

```text
main
```

para armazenar a versão final do projeto.

Durante o desenvolvimento, foi utilizada:

```text
develop
```

como branch de integração.

As funcionalidades foram desenvolvidas em feature branches, incluindo:

```text
feature/eda
feature/preprocessamento
feature/modelagem
feature/avaliacao
```

O fluxo utilizado foi:

```text
feature/*
    │
    ▼
develop
    │
    ▼
main
```

As alterações foram integradas por meio de Pull Requests.

As feature branches foram mantidas após os merges para preservar o histórico de desenvolvimento do projeto.

---

# 24. Como Executar o Projeto

Clone o repositório:

```bash
git clone <URL-DO-REPOSITORIO>
```

Entre na pasta:

```bash
cd SENAI-Miniprojeto-MNIST
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o projeto no Jupyter Notebook ou VS Code e execute o notebook principal.

---

# 25. Reprodutibilidade

Sempre que aplicável, foi utilizado:

```python
RANDOM_STATE = 42
```

para tornar os experimentos mais reprodutíveis.

Resultados relacionados ao tempo de execução podem variar de acordo com o hardware e as condições do ambiente utilizado.

---

# 26. Autor

Projeto desenvolvido como atividade de Machine Learning do **SENAI**.

**Projeto:** Classificação de Dígitos Manuscritos — MNIST
