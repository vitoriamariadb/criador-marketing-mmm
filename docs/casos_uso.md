# Casos de Uso do Marketing Mix Model

## Caso 1: Otimizacao de Budget Trimestral

### Contexto
Empresa de e-commerce com investimento mensal de R$500.000 em 6 canais de midia.
Deseja encontrar a melhor alocacao para o proximo trimestre.

### Passos

1. Carregar dados de investimento e receita semanal dos ultimos 2 anos
2. Aplicar adstock geometrico com decay otimizado
3. Aplicar saturacao exponencial
4. Treinar modelo Ridge com validacao cruzada temporal
5. Executar otimizacao de budget com restricoes

### Restricoes Tipicas
- Minimo de 10% do budget por canal (diversificacao)
- Maximo de 40% em um unico canal
- TV tem compromisso de compra de midia (minimo R$100.000)

### Resultado Esperado
Alocacao otima por canal com receita prevista e ROAS esperado.

---

## Caso 2: Avaliacao de Novo Canal

### Contexto
Equipe de marketing quer avaliar se o investimento em podcasts (novo canal)
esta gerando retorno adequado apos 6 meses de teste.

### Passos

1. Incluir investimento em podcast como nova variavel
2. Retreinar modelo com dados atualizados
3. Analisar coeficiente e significancia do novo canal
4. Comparar ROAS com outros canais
5. Analisar curva de saturacao para identificar potencial

### Criterios de Avaliacao
- ROAS > 1.5x (minimo para justificar continuidade)
- Coeficiente positivo e estavel na validacao cruzada
- Canal nao saturado (potencial de crescimento)

---

## Caso 3: Simulacao de Corte de Budget

### Contexto
Cenario de crise economica. Budget de marketing sera reduzido em 30%.
Precisa minimizar o impacto na receita.

### Passos

1. Criar cenario baseline com alocacao atual
2. Criar cenario de corte proporcional (-30% em todos)
3. Criar cenario otimizado (corte de 30% com realocacao inteligente)
4. Comparar receita prevista dos cenarios

### Estrategia de Realocacao
1. Identificar canais mais saturados (menor ROAS marginal)
2. Cortar mais nos canais saturados
3. Manter investimento nos canais com maior eficiencia marginal

---

## Caso 4: Planejamento Sazonal

### Contexto
Planejar alocacao diferenciada para Black Friday e Natal (Q4).

### Passos

1. Analisar efeito da sazonalidade no modelo
2. Simular cenarios de aumento de budget no Q4
3. Avaliar diminishing returns por canal no periodo sazonal
4. Definir alocacao diferenciada para semanas de pico

### Consideracoes
- Adstock durante periodos de pico pode ter decays diferentes
- Competidores tambem aumentam investimento (considerar competitor_spend)
- Holiday_flag captura efeito geral, mas cenarios especificos requerem ajuste

---

## Caso 5: Report Executivo Mensal

### Contexto
Gerar relatorio mensal de performance de marketing para a diretoria.

### Conteudo do Relatorio

1. **Resumo executivo**: receita total, investimento total, ROAS geral
2. **Decomposicao de receita**: contribuicao de cada canal (waterfall chart)
3. **ROAS por canal**: ranking de eficiencia
4. **Tendencias**: evolucao temporal das contribuicoes
5. **Recomendacoes**: canais para aumentar/reduzir investimento

### Exportacao
- Dashboard interativo (Streamlit) para exploracao
- PDF/Excel para apresentacao executiva
- JSON via API para integracao com sistemas internos

---

## Caso 6: Integracao com Planejamento de Midia

### Contexto
Integrar resultados do MMM com a plataforma de planejamento de midia da agencia.

### Fluxo de Integracao

1. **API /train**: treina modelo com dados mais recentes
2. **API /coefficients**: fornece pesos de cada canal
3. **API /optimize**: recebe budget e retorna alocacao otima
4. **API /scenarios**: simula cenarios propostos pela agencia

### Automacao
- Cron job semanal para retreinar modelo com novos dados
- Webhook para notificar equipe quando metricas mudam significativamente
- Dashboard atualizado em tempo real via Streamlit

---

## Caso 7: Analise Cross-Channel

### Contexto
Entender como a midia offline (TV, radio) amplifica a performance digital.

### Passos

1. Incluir features de interacao (TV x Digital, TV x Search)
2. Analisar coeficientes de interacao
3. Se positivos, existe efeito de amplificacao cruzada
4. Quantificar o "halo effect" da TV sobre digital

### Interpretacao
- Interacao positiva: TV gera awareness que amplifica busca e conversao digital
- Interacao negativa: canais competem pelo mesmo publico (substituicao)
- Sem interacao significativa: canais operam de forma independente

---

*"A teoria sem pratica e inutil, a pratica sem teoria e perigosa." - Confucio*
