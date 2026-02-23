# Metodologia de Marketing Mix Modeling (MMM)

## 1. Visão Geral

Marketing Mix Modeling (MMM) é uma técnica estatística que quantifica o impacto de
diferentes canais de marketing na receita ou outra variável de negócio. Diferente de
modelos de atribuição digital (last-click, multi-touch), o MMM funciona com dados
agregados e consegue capturar efeitos offline (TV, rádio, mídia impressa).

## 2. Fundamentação Teórica

### 2.1 Modelo Base

O modelo base é uma regressão linear múltipla:

```
Revenue = b0 + b1*X1 + b2*X2 + ... + bn*Xn + e
```

Onde:
- `b0`: intercepto (receita base sem marketing)
- `Xi`: variáveis de investimento em mídia ou controle
- `bi`: coeficientes que representam o efeito marginal
- `e`: erro aleatório

### 2.2 Transformação Adstock

A publicidade tem efeito residual - o impacto de um anúncio não desaparece imediatamente.
O adstock modela esse carry-over:

```
Adstock_t = X_t + decay * Adstock_{t-1}
```

Parâmetros:
- **decay_rate**: controla a velocidade do decaimento (0 = sem efeito residual, 1 = efeito permanente)
- **max_lag**: número máximo de períodos de carry-over

Tipos de adstock implementados:
1. **Geométrico**: decaimento exponencial simples
2. **Weibull**: permite pico atrasado (delayed peak effect)

### 2.3 Curvas de Saturação

Retornos decrescentes são modelados por curvas de saturação. Após certo nível de investimento,
o retorno marginal diminui.

Funções implementadas:
1. **Exponencial**: `y = 1 - exp(-lambda * x)` - saturação monótona
2. **Hill**: `y = x^a / (x^a + g^a)` - função sigmoide flexível
3. **Logística**: curva S clássica com ponto de inflexão
4. **Potência**: `y = x^a` onde a < 1 produz retornos decrescentes

### 2.4 Regularização

Modelos regularizados previnem overfitting:

- **Ridge (L2)**: penaliza a soma dos quadrados dos coeficientes. Todos permanecem não-zero.
- **Lasso (L1)**: penaliza a soma dos valores absolutos. Promove sparsidade (alguns coeficientes vão a zero).
- **ElasticNet**: combinação de L1 e L2. Parâmetro `l1_ratio` controla o mix.

## 3. Pipeline de Modelagem

### 3.1 Ingestão de Dados

- Dados semanais ou diários de investimento por canal
- Variável dependente (receita, conversões, etc.)
- Variáveis de controle (sazonalidade, preço, competidores)

### 3.2 Feature Engineering

1. Features temporais (semana, mês, sazonalidade cíclica)
2. Transformação adstock nos canais de mídia
3. Transformação de saturação
4. Lags e médias móveis
5. Features de share of spend
6. Interações entre canais (opcional)

### 3.3 Seleção e Treinamento

1. Comparação entre Ridge, Lasso e ElasticNet
2. Validação cruzada temporal (TimeSeriesSplit)
3. Grid search para hiperparâmetros
4. Avaliação: R2, RMSE, MAE, MAPE

### 3.4 Interpretação

1. Coeficientes do modelo = efeito marginal de cada canal
2. Decomposição de contribuição = quanto cada canal contribuiu para a receita
3. ROI/ROAS por canal = eficiência do investimento
4. Feature importance por permutação

## 4. Otimização de Budget

A otimização busca a alocação que maximiza a receita prevista, sujeita a:
- Orçamento total fixo
- Limites mínimos e máximos por canal
- Curvas de saturação (retornos decrescentes)

Método: scipy.optimize.minimize com restrições (SLSQP).

## 5. Planejamento de Cenários

Tipos de cenários suportados:
1. **Proporcional**: aumento/redução uniforme em todos os canais
2. **Foco em canal**: aumento específico em um canal
3. **Realocação**: transferência de budget entre canais
4. **Análise de sensibilidade**: impacto de variações em um canal

## 6. Limitações do MMM

- Requer dados históricos suficientes (idealmente 2+ anos)
- Assume relação linear (ou linearizada via transformações)
- Não captura interações complexas entre canais
- Sensível a multicolinearidade entre canais
- Resultados dependem da qualidade das transformações (adstock, saturação)

## 7. Referências

- Jin, Y., et al. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects"
- Chan, D., & Perry, M. (2017). "Challenges and Opportunities in Media Mix Modeling"
- Google (2017). "Media Mix Model - Bayesian Regression"
- Robyn by Meta (2022). Documentação técnica de referência

---

*"A estatística é a gramática da ciência." - Karl Pearson*
