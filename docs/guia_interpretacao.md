# Guia de Interpretação dos Resultados do MMM

## 1. Como Ler os Coeficientes

Os coeficientes do modelo representam o efeito marginal de cada variável na receita.

### 1.1 Coeficiente positivo
Um coeficiente positivo indica que o aumento do investimento naquele canal está
associado a um aumento na receita.

**Exemplo**: Se o coeficiente de `tv_spend` é 1.2, para cada R$1.000 adicional
investido em TV (após transformações de adstock e saturação), a receita tende a
aumentar em R$1.200 (em escala padronizada).

### 1.2 Coeficiente negativo
Pode indicar que o canal não está gerando retorno ou que há multicolinearidade.

**Atenção**: coeficientes negativos em canais de mídia devem ser investigados.
Possíveis causas:
- Multicolinearidade entre canais
- Investimento concentrado em períodos de baixa demanda
- Volume de dados insuficiente
- Especificação incorreta do modelo

### 1.3 Importância relativa
A importância relativa é calculada como: |coeficiente_i| / soma(|todos coeficientes|) * 100

Quanto maior a importância relativa, maior o impacto do canal na variância da receita.

## 2. Como Interpretar a Decomposição de Receita

A decomposição mostra quanto cada canal contribuiu para a receita total no período.

### 2.1 Base (Intercepto)
Representa a receita que ocorreria sem nenhum investimento em marketing.
Inclui demanda orgânica, brand equity histórico e fatores não modelados.

### 2.2 Contribuição por canal
O valor de contribuição de cada canal indica quanto da receita é atribuível ao investimento
naquele canal, considerando adstock e saturação.

**Leitura**: Se o canal "digital" contribuiu com R$150.000 no período e o investimento foi
R$50.000, o ROAS é 3.0x (para cada R$1 investido, retornaram R$3).

## 3. Como Interpretar ROI e ROAS

### 3.1 ROAS (Return on Ad Spend)
```
ROAS = Receita atribuída ao canal / Investimento no canal
```

- ROAS > 1: o canal gera mais receita do que custa
- ROAS = 1: ponto de equilíbrio
- ROAS < 1: o canal custa mais do que gera

### 3.2 ROI (Return on Investment)
```
ROI = (Receita atribuída - Investimento) / Investimento * 100
```

- ROI > 0%: retorno positivo
- ROI = 0%: break-even
- ROI < 0%: retorno negativo

### 3.3 Cuidados na interpretação
- ROAS alto com investimento baixo pode não escalar
- ROAS deve ser analisado junto com a curva de saturação
- Compare ROAS com benchmarks do setor

## 4. Como Interpretar Curvas de Saturação

### 4.1 Canal não saturado
Curva ainda na região linear (início). Investimento adicional terá retorno proporcional.
**Ação**: considerar aumento de investimento.

### 4.2 Canal parcialmente saturado
Curva na região de transição. Investimento adicional tem retorno, mas decrescente.
**Ação**: manter ou otimizar alocação.

### 4.3 Canal altamente saturado
Curva no plateau. Investimento adicional terá retorno marginal muito baixo.
**Ação**: considerar realocação para canais menos saturados.

## 5. Como Interpretar Cenários

### 5.1 Delta vs Baseline
Mostra a diferença percentual na receita prevista em relação ao cenário atual.

**Exemplo**: Se o cenário "Aumento 20% Digital" mostra delta de +5%, significa que
aumentar o investimento digital em 20% (mantendo outros canais iguais) deve aumentar
a receita total em 5%.

### 5.2 Cenários de realocação
Quando o orçamento total é fixo, a realocação de canais saturados para canais com maior
retorno marginal tende a maximizar a receita total.

## 6. Métricas do Modelo

### 6.1 R-quadrado (R2)
Proporção da variância da receita explicada pelo modelo.
- R2 > 0.90: muito bom
- 0.80 < R2 < 0.90: bom
- 0.70 < R2 < 0.80: aceitável
- R2 < 0.70: revisar especificação

### 6.2 MAPE (Mean Absolute Percentage Error)
Erro médio percentual absoluto.
- MAPE < 5%: excelente
- 5% < MAPE < 10%: bom
- 10% < MAPE < 20%: aceitável
- MAPE > 20%: revisar modelo

### 6.3 R-quadrado Ajustado
Penaliza o R2 pelo número de features. Útil para comparar modelos com
diferentes quantidades de variáveis.

## 7. Red Flags

Sinais de que algo pode estar errado nos resultados:

1. **Coeficientes negativos em canais de mídia com ROAS histórico positivo**
2. **R2 muito alto (> 0.99)** - possível overfitting
3. **Grandes diferenças entre treino e teste** - instabilidade do modelo
4. **ROAS irrealisticamente alto** - verificar escala e transformações
5. **Base (intercepto) negativa** - modelo mal especificado

---

*"Os números não mentem, mas mentirosos usam números." - Mark Twain*
