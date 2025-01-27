# Aula 1

### sklearn.model_selection.train_test_split

O train_test_split gera essas quatro variáveis para simular um cenário real, onde o modelo será apresentado a novos dados que ele não conhece. Ao comparar as previsões do modelo com os valores reais no conjunto de teste, podemos avaliar a sua qualidade e identificar possíveis problemas, como overfitting ou underfitting.

Exemplo:

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
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.

- train_size

  - float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.

- random_state

  - int, RandomState instance or None, default=None
    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. See Glossary.

- shuffle

  - bool, default=True
    Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

- stratify
  - array-like, default=None
    If not None, data is split in a stratified fashion, using this as the class labels. Read more in the User Guide.

---

## Explicações do código

- Preparação do dataset

```python
# Variáveis independentes (X) e dependente (y)
X = df[['size', 'bedrooms']]
y = df['price']

# Dividir em treino e teste
# Explicação do metodo dentro do readme.md
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**O que é?**
No Scikit-learn , ele controla o `train_test_split` embaralhamento aplicado aos dados antes de aplicar a divisão. Nós o usamos `train_test_split` para dividir dados em conjuntos de dados de treinamento e teste. Ele assume um dos seguintes valores.

_None_ (Padrão)
Ele usa a instância global de estado aleatório de `numpy.random`. Se chamarmos a mesma função com `random_state=none` então ela produzirá resultados diferentes em cada execução.
_Int_
Se usarmos qualquer valor inteiro para `random_state` então ele produzirá o mesmo resultado para um valor inteiro. Se mudarmos o valor de `random_state` , então somente o resultado será diferente.

**`random_state` não pode ser negativo!!!**

**Parametros usados**

- X, y

  - São os arrays a serem serapados sendo `X` as variáveis independentes (A serem "Aprendidas") e `y` a variável dependente (A ser "Prevista").
    Ou seja nesse caso o valor do imovel seria diretamente impactado pela metragem do imovel o numero de quartos.

- test_size=0.2

  - Se for um valor entre 0 e 1 define a porcentagem do banco a ser considerado treino, nesse caso 20% do banco será considerado treino. Caso seja um numero _`int`_ o numero fornecido será o numero de amostras a serem consideradas como treino. Ex um teste*size = 20 significa que independente do tamanho do \_dataset* apenas 20 amostras serão consideradas como treino.

- random_state=42
  - Define os estados de aleatoriedade do `dataset` por exemplo:
    Partindo do _dataset_ inicial, seriam realizados 42 embaralhamentos. E por que disso?
    Um conjunto de dados que não foi embaralhado pode gerar problemas de predição uma vez que os dados não foram mesclados da melhor maneiro criando assim uma grande chance de vies nos dados. E nesse caso utilizando o dataset nas mesmas condições com o mesmo valor de _random_state_ chegaremos sempre no mesmo resultado, facilitando a reprodutibilidade e a consistencia do treinamento entre os testes.
    No entanto há um limite de combinações possiveis, o número de combições possiveis pode ser calculado pela formula: C=(n,r)
    $$ncr = \frac{N!}{R! * (N - R)!}$$
    Então para um para um _dataset_ com 10 elementos sendo 7 de treino e 3 de teste temos:
    $$ncr = \frac{10!}{3! * (10 - 3)!} = \frac{10!}{3! * 7!}$$
    $$ncr = \frac{10.9.8.7.6.5.4.3.2.1}{(3.2.1) * (7.6.5.4.3.2.1)} = \frac{3628800}{30240} = 120$$
    Sendo assim existiem apenas 120 combinações possíveis para esse conjunto, logo o `random_state` deve ser algo entre 0 e 119.

---

```python
model = LinearRegression()
```

### Regressão linear

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
  y = Variável dependente (altura)
  m = Inclinação da reta
  x = Variável independente (idade)
  b = Intercepto (ponto onde a reta cruza o eixo y)

---

### Fit function

```python
model.fit(X_train, y_train)
```

Essa função realiza o treinamento do modelo usando o `X_train` que no código representa as variáveis independentes ou seja, que infuenciam no valor da casa ex numero de quartos e metragem, o valor dessas casas de treino é armazenada na variável `y_train`.

---

### Predições

Gera a predição dos dados no conjunto de teste

```python
# Previsões no conjunto de teste
y_pred = model.predict(X_test)
```

**A grosso modo**
A função pega os valores independentes e pensa Ok, $60m^2$ e 2 quartos (`X_train`) está saindo por 160000 (`y_train`). Logo uma casa com $90m^2$ e 3 quartos (`X_test`) deve custar $160000*1.5$ (`predição`).

---

**Calcula Erro Quadrático Médio (MSE)**

```python
# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio (MSE):", mse)
```

O Mean Squared Error (MSE), ou Erro Quadrático Médio em português, é uma métrica amplamente utilizada em machine learning, especialmente em problemas de regressão, para avaliar a performance de um modelo. Ele quantifica a média dos erros quadráticos entre os valores previstos pelo modelo e os valores reais.

**Por que usar o MSE?**

Simplicidade: É fácil de calcular e entender.
Punição de erros grandes: Ao elevar os erros ao quadrado, o MSE penaliza mais os erros maiores, dando-lhes mais peso na avaliação final. Isso significa que o modelo será "penalizado" mais severamente por previsões muito distantes dos valores reais.
Escalabilidade: Pode ser usado para avaliar a performance de modelos em diferentes escalas.
Como o MSE é calculado?

**Cálculo dos erros:**
Para cada observação, calcula-se a diferença entre o valor previsto pelo modelo e o valor real.
Elevação ao quadrado: Cada erro calculado é elevado ao quadrado. Isso garante que todos os erros sejam positivos e que os erros maiores tenham um impacto maior no resultado final.
Cálculo da média: A média de todos os erros quadrados é calculada.
Formula:

$$MSE = (\frac{1}{n}) * Σ(y_i - ŷ_i)^2$$
Onde:

n: número de observações
y_i: valor real da i-ésima observação
ŷ_i: valor previsto pela i-ésima observação
Interpretação:

**Valor baixo:** Indica que o modelo está fazendo previsões precisas, com erros pequenos.
**Valor alto:** Indica que o modelo está fazendo previsões imprecisas, com erros grandes.
Exemplo:

Imagine que você tenha um modelo que prediz a altura de pessoas com base em sua idade. Se o modelo prever que uma pessoa com 20 anos tenha 180 cm, mas na verdade ela tenha 175 cm, o erro para essa observação seria 5 cm. Ao elevar ao quadrado e calcular a média de todos os erros, obtém-se o MSE.

**Qual a relação do MSE com outras métricas?**

RMSE (Root Mean Squared Error): É a raiz quadrada do MSE. A vantagem do RMSE é que ele tem a mesma unidade de medida da variável alvo, facilitando a interpretação.
MAE (Mean Absolute Error): Calcula a média dos valores absolutos dos erros. É menos sensível a outliers do que o MSE.
Quando usar o MSE?
O MSE é uma métrica muito útil, mas é importante lembrar que ele tem suas limitações. Ele é mais adequado para problemas onde erros grandes são mais penalizantes. Em outras situações, como quando você tem muitos outliers, o MAE pode ser uma melhor opção.

---

**Informações adicionais**

```python
# Exibir os coeficientes do modelo
print("Intercepto (β₀):", model.intercept_)
print("Coeficientes (β₁, β₂):", model.coef_)
```

### Exibição dos Coeficientes

`intercept_`: Imprime o intercepto da reta de regressão. Representa o preço médio de uma casa que não possui quartos e tem tamanho zero. Na prática, esse valor pode não ter significado físico, mas é necessário para a equação da reta.
`coef_`: Imprime os coeficientes da reta de regressão, que representam a influência de cada variável independente (size e bedrooms) sobre o preço da casa. Por exemplo, se o coeficiente da variável 'size' for 1000, significa que, em média, o preço da casa aumenta R$1000 para cada metro quadrado adicional.
