# Metodologia de Marketing Mix Modeling (MMM)

## 1. Visao Geral

Marketing Mix Modeling (MMM) e uma tecnica estatistica que quantifica o impacto de
diferentes canais de marketing na receita ou outra variavel de negocio. Diferente de
modelos de atribuicao digital (last-click, multi-touch), o MMM funciona com dados
agregados e consegue capturar efeitos offline (TV, radio, midia impressa).

## 2. Fundamentacao Teorica

### 2.1 Modelo Base

O modelo base e uma regressao linear multipla:

```
Revenue = b0 + b1*X1 + b2*X2 + ... + bn*Xn + e
```

Onde:
- `b0`: intercepto (receita base sem marketing)
- `Xi`: variaveis de investimento em midia ou controle
- `bi`: coeficientes que representam o efeito marginal
- `e`: erro aleatorio

### 2.2 Transformacao Adstock

A publicidade tem efeito residual - o impacto de um anuncio nao desaparece imediatamente.
O adstock modela esse carry-over:

```
Adstock_t = X_t + decay * Adstock_{t-1}
```

Parametros:
- **decay_rate**: controla a velocidade do decaimento (0 = sem efeito residual, 1 = efeito permanente)
- **max_lag**: numero maximo de periodos de carry-over

Tipos de adstock implementados:
1. **Geometrico**: decaimento exponencial simples
2. **Weibull**: permite pico atrasado (delayed peak effect)

### 2.3 Curvas de Saturacao

Retornos decrescentes sao modelados por curvas de saturacao. Apos certo nivel de investimento,
o retorno marginal diminui.

Funcoes implementadas:
1. **Exponencial**: `y = 1 - exp(-lambda * x)` - saturacao monotona
2. **Hill**: `y = x^a / (x^a + g^a)` - funcao sigmoide flexivel
3. **Logistica**: curva S classica com ponto de inflexao
4. **Potencia**: `y = x^a` onde a < 1 produz retornos decrescentes

### 2.4 Regularizacao

Modelos regularizados previnem overfitting:

- **Ridge (L2)**: penaliza a soma dos quadrados dos coeficientes. Todos permanecem nao-zero.
- **Lasso (L1)**: penaliza a soma dos valores absolutos. Promove sparsidade (alguns coeficientes vao a zero).
- **ElasticNet**: combinacao de L1 e L2. Parametro `l1_ratio` controla o mix.

## 3. Pipeline de Modelagem

### 3.1 Ingestao de Dados

- Dados semanais ou diarios de investimento por canal
- Variavel dependente (receita, conversoes, etc.)
- Variaveis de controle (sazonalidade, preco, competidores)

### 3.2 Feature Engineering

1. Features temporais (semana, mes, sazonalidade ciclica)
2. Transformacao adstock nos canais de midia
3. Transformacao de saturacao
4. Lags e medias moveis
5. Features de share of spend
6. Interacoes entre canais (opcional)

### 3.3 Selecao e Treinamento

1. Comparacao entre Ridge, Lasso e ElasticNet
2. Validacao cruzada temporal (TimeSeriesSplit)
3. Grid search para hiperparametros
4. Avaliacao: R2, RMSE, MAE, MAPE

### 3.4 Interpretacao

1. Coeficientes do modelo = efeito marginal de cada canal
2. Decomposicao de contribuicao = quanto cada canal contribuiu para a receita
3. ROI/ROAS por canal = eficiencia do investimento
4. Feature importance por permutacao

## 4. Otimizacao de Budget

A otimizacao busca a alocacao que maximiza a receita prevista, sujeita a:
- Orcamento total fixo
- Limites minimos e maximos por canal
- Curvas de saturacao (retornos decrescentes)

Metodo: scipy.optimize.minimize com restricoes (SLSQP).

## 5. Planejamento de Cenarios

Tipos de cenarios suportados:
1. **Proporcional**: aumento/reducao uniforme em todos os canais
2. **Foco em canal**: aumento especifico em um canal
3. **Realocacao**: transferencia de budget entre canais
4. **Analise de sensibilidade**: impacto de variacoes em um canal

## 6. Limitacoes do MMM

- Requer dados historicos suficientes (idealmente 2+ anos)
- Assume relacao linear (ou linearizada via transformacoes)
- Nao captura interacoes complexas entre canais
- Sensivel a multicolinearidade entre canais
- Resultados dependem da qualidade das transformacoes (adstock, saturacao)

## 7. Referencias

- Jin, Y., et al. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects"
- Chan, D., & Perry, M. (2017). "Challenges and Opportunities in Media Mix Modeling"
- Google (2017). "Media Mix Model - Bayesian Regression"
- Robyn by Meta (2022). Documentacao tecnica de referencia

---

*"A estatistica e a gramatica da ciencia." - Karl Pearson*
