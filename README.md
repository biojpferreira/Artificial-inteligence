# Inteligencia artificial

- [Inteligencia artificial](#inteligencia-artificial)
  - [Conceitos](#conceitos)
    - [_overfitting_ e _underfitting_](#-overfitting-e-underfitting)
  - [Bibliotecas e funções](#bibliotecas-e-fun-es)
    - [sklearn.model_selection.train_test_split](#sklearnmodel-selectiontrain-test-split)
  - [Modelos](#modelos)
    - [Regressão Linear (Linear Regression)](#regress-o-linear-linear-regression)

## Conceitos

### _overfitting_ e _underfitting_

## Bibliotecas e funções

### sklearn.model_selection.train_test_split

O train*test_split gera essas quatro variáveis para simular um cenário real, onde o modelo será apresentado a novos dados que ele não conhece. Ao comparar as previsões do modelo com os valores reais no conjunto de teste, podemos avaliar a sua qualidade e identificar possíveis problemas, como \_overfitting* ou _underfitting_.

**Exemplo:**

Imagine que você quer criar um modelo para prever o preço de uma casa com base em suas características (número de quartos, área, localização, etc.). Você teria um conjunto de dados com várias casas, onde cada casa é uma observação e as características são as features (X). O preço da casa seria o target (y).

Ao aplicar o train_test_split, você dividiria esses dados em:

X_train: Características de um conjunto de casas que serão usadas para treinar o modelo.
X_test: Características de outro conjunto de casas que serão usadas para testar o modelo.
y_train: Preços das casas no conjunto de treinamento.
y_test: Preços das casas no conjunto de teste.
O modelo seria treinado usando X_train e y_train, e depois seria utilizado para fazer previsões nos dados de X_test. As previsões seriam então comparadas com os valores reais em y_test para avaliar a performance do modelo.

**Parameters:**

- \*arrays

- sequence of indexables with same length / shape[0].
  Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.

- test_size

  - float or int, default=None
    Se for um valor entre 0 e 1 define a porcentagem do banco a ser considerado treino, nesse caso 20% do banco será considerado treino. Caso seja um numero _`int`_ o numero fornecido será o numero de amostras a serem consideradas como treino. Ex um teste*size = 20 significa que independente do tamanho do \_dataset* apenas 20 amostras serão consideradas como treino.

- train_size

  - float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.

- random_state

  - Define os estados de aleatoriedade do `dataset` por exemplo:
    Partindo do _dataset_ inicial, seriam realizados 42 embaralhamentos. E por que disso?
    Um conjunto de dados que não foi embaralhado pode gerar problemas de predição uma vez que os dados não foram mesclados da melhor maneiro criando assim uma grande chance de vies nos dados. E nesse caso utilizando o dataset nas mesmas condições com o mesmo valor de _random_state_ chegaremos sempre no mesmo resultado, facilitando a reprodutibilidade e a consistencia do treinamento entre os testes.
    No entanto há um limite de combinações possiveis, o número de combições possiveis pode ser calculado pela formula: C=(n,r)
    $$ncr = \frac{N!}{R! * (N - R)!}$$
    Então para um para um _dataset_ com 10 elementos sendo 7 de treino e 3 de teste temos:
    $$ncr = \frac{10!}{3! * (10 - 3)!} = \frac{10!}{3! * 7!}$$
    $$ncr = \frac{10.9.8.7.6.5.4.3.2.1}{(3.2.1) * (7.6.5.4.3.2.1)} = \frac{3628800}{30240} = 120$$
    Sendo assim existiem apenas 120 combinações possíveis para esse conjunto, logo o `random_state` deve ser algo entre 0 e 119.

- shuffle

  - bool, default=True
    Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

- stratify
  - array-like, default=None
    If not None, data is split in a stratified fashion, using this as the class labels. Read more in the User Guide.

---

## Modelos

### Regressão Linear (Linear Regression)

**O que é Regressão Linear?**
Imagine que você queira prever a altura de uma pessoa com base em sua idade. Ou talvez prever o preço de uma casa com base em sua área. A regressão linear é uma ferramenta estatística que nos ajuda a encontrar a relação entre duas variáveis, como altura e idade, ou preço e área.

Em termos mais técnicos, a regressão linear busca encontrar a melhor linha reta que se ajuste a um conjunto de dados. Essa linha reta representa a relação entre as duas variáveis. A equação dessa linha é chamada de equação de regressão.

**Outras técnicas de regressão:**

Além da regressão linear, existem outras técnicas de regressão, como a regressão logística (para variáveis dependentes binárias), a regressão polinomial (para relações não lineares), entre outras. A escolha da técnica adequada depende do tipo de dados e do problema que você está tentando resolver.

**Por que usar a regressão linear?**

- Previsão: Prever valores futuros de uma variável com base em outra.
- Entendimento: Compreender a força e a direção da relação entre duas variáveis.
- Tomada de decisão: Tomar decisões baseadas em modelos preditivos.

**Equação**
A equação geral de uma reta é:
$$y = mx + b$$

- Onde
  - y = Variável dependente (altura)
  - m = Inclinação da reta
  - x = Variável independente (idade)
  - b = Intercepto (ponto onde a reta cruza o eixo y)

---
