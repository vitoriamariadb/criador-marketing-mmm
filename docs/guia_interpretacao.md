# Guia de Interpretacao dos Resultados do MMM

## 1. Como Ler os Coeficientes

Os coeficientes do modelo representam o efeito marginal de cada variavel na receita.

### 1.1 Coeficiente positivo
Um coeficiente positivo indica que o aumento do investimento naquele canal esta
associado a um aumento na receita.

**Exemplo**: Se o coeficiente de `tv_spend` e 1.2, para cada R$1.000 adicional
investido em TV (apos transformacoes de adstock e saturacao), a receita tende a
aumentar em R$1.200 (em escala padronizada).

### 1.2 Coeficiente negativo
Pode indicar que o canal nao esta gerando retorno ou que ha multicolinearidade.

**Atencao**: coeficientes negativos em canais de midia devem ser investigados.
Possiveis causas:
- Multicolinearidade entre canais
- Investimento concentrado em periodos de baixa demanda
- Volume de dados insuficiente
- Especificacao incorreta do modelo

### 1.3 Importancia relativa
A importancia relativa e calculada como: |coeficiente_i| / soma(|todos coeficientes|) * 100

Quanto maior a importancia relativa, maior o impacto do canal na variancia da receita.

## 2. Como Interpretar a Decomposicao de Receita

A decomposicao mostra quanto cada canal contribuiu para a receita total no periodo.

### 2.1 Base (Intercepto)
Representa a receita que ocorreria sem nenhum investimento em marketing.
Inclui demanda organica, brand equity historico e fatores nao modelados.

### 2.2 Contribuicao por canal
O valor de contribuicao de cada canal indica quanto da receita e atribuivel ao investimento
naquele canal, considerando adstock e saturacao.

**Leitura**: Se o canal "digital" contribuiu com R$150.000 no periodo e o investimento foi
R$50.000, o ROAS e 3.0x (para cada R$1 investido, retornaram R$3).

## 3. Como Interpretar ROI e ROAS

### 3.1 ROAS (Return on Ad Spend)
```
ROAS = Receita atribuida ao canal / Investimento no canal
```

- ROAS > 1: o canal gera mais receita do que custa
- ROAS = 1: ponto de equilibrio
- ROAS < 1: o canal custa mais do que gera

### 3.2 ROI (Return on Investment)
```
ROI = (Receita atribuida - Investimento) / Investimento * 100
```

- ROI > 0%: retorno positivo
- ROI = 0%: break-even
- ROI < 0%: retorno negativo

### 3.3 Cuidados na interpretacao
- ROAS alto com investimento baixo pode nao escalar
- ROAS deve ser analisado junto com a curva de saturacao
- Compare ROAS com benchmarks do setor

## 4. Como Interpretar Curvas de Saturacao

### 4.1 Canal nao saturado
Curva ainda na regiao linear (inicio). Investimento adicional tera retorno proporcional.
**Acao**: considerar aumento de investimento.

### 4.2 Canal parcialmente saturado
Curva na regiao de transicao. Investimento adicional tem retorno, mas decrescente.
**Acao**: manter ou otimizar alocacao.

### 4.3 Canal altamente saturado
Curva no plateau. Investimento adicional tera retorno marginal muito baixo.
**Acao**: considerar realocacao para canais menos saturados.

## 5. Como Interpretar Cenarios

### 5.1 Delta vs Baseline
Mostra a diferenca percentual na receita prevista em relacao ao cenario atual.

**Exemplo**: Se o cenario "Aumento 20% Digital" mostra delta de +5%, significa que
aumentar o investimento digital em 20% (mantendo outros canais iguais) deve aumentar
a receita total em 5%.

### 5.2 Cenarios de realocacao
Quando o orcamento total e fixo, a realocacao de canais saturados para canais com maior
retorno marginal tende a maximizar a receita total.

## 6. Metricas do Modelo

### 6.1 R-quadrado (R2)
Proporcao da variancia da receita explicada pelo modelo.
- R2 > 0.90: muito bom
- 0.80 < R2 < 0.90: bom
- 0.70 < R2 < 0.80: aceitavel
- R2 < 0.70: revisar especificacao

### 6.2 MAPE (Mean Absolute Percentage Error)
Erro medio percentual absoluto.
- MAPE < 5%: excelente
- 5% < MAPE < 10%: bom
- 10% < MAPE < 20%: aceitavel
- MAPE > 20%: revisar modelo

### 6.3 R-quadrado Ajustado
Penaliza o R2 pelo numero de features. Util para comparar modelos com
diferentes quantidades de variaveis.

## 7. Red Flags

Sinais de que algo pode estar errado nos resultados:

1. **Coeficientes negativos em canais de midia com ROAS historico positivo**
2. **R2 muito alto (> 0.99)** - possivel overfitting
3. **Grandes diferencas entre treino e teste** - instabilidade do modelo
4. **ROAS irrealisticamente alto** - verificar escala e transformacoes
5. **Base (intercepto) negativa** - modelo mal especificado

---

*"Os numeros nao mentem, mas mentirosos usam numeros." - Mark Twain*
