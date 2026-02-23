# Casos de Uso do Marketing Mix Model

## Caso 1: Otimização de Budget Trimestral

### Contexto
Empresa de e-commerce com investimento mensal de R$500.000 em 6 canais de mídia.
Deseja encontrar a melhor alocação para o próximo trimestre.

### Passos

1. Carregar dados de investimento e receita semanal dos últimos 2 anos
2. Aplicar adstock geométrico com decay otimizado
3. Aplicar saturação exponencial
4. Treinar modelo Ridge com validação cruzada temporal
5. Executar otimização de budget com restrições

### Restrições Típicas
- Mínimo de 10% do budget por canal (diversificação)
- Máximo de 40% em um único canal
- TV tem compromisso de compra de mídia (mínimo R$100.000)

### Resultado Esperado
Alocação ótima por canal com receita prevista e ROAS esperado.

---

## Caso 2: Avaliação de Novo Canal

### Contexto
Equipe de marketing quer avaliar se o investimento em podcasts (novo canal)
está gerando retorno adequado após 6 meses de teste.

### Passos

1. Incluir investimento em podcast como nova variável
2. Retreinar modelo com dados atualizados
3. Analisar coeficiente e significância do novo canal
4. Comparar ROAS com outros canais
5. Analisar curva de saturação para identificar potencial

### Critérios de Avaliação
- ROAS > 1.5x (mínimo para justificar continuidade)
- Coeficiente positivo e estável na validação cruzada
- Canal não saturado (potencial de crescimento)

---

## Caso 3: Simulação de Corte de Budget

### Contexto
Cenário de crise econômica. Budget de marketing será reduzido em 30%.
Precisa minimizar o impacto na receita.

### Passos

1. Criar cenário baseline com alocação atual
2. Criar cenário de corte proporcional (-30% em todos)
3. Criar cenário otimizado (corte de 30% com realocação inteligente)
4. Comparar receita prevista dos cenários

### Estratégia de Realocação
1. Identificar canais mais saturados (menor ROAS marginal)
2. Cortar mais nos canais saturados
3. Manter investimento nos canais com maior eficiência marginal

---

## Caso 4: Planejamento Sazonal

### Contexto
Planejar alocação diferenciada para Black Friday e Natal (Q4).

### Passos

1. Analisar efeito da sazonalidade no modelo
2. Simular cenários de aumento de budget no Q4
3. Avaliar diminishing returns por canal no período sazonal
4. Definir alocação diferenciada para semanas de pico

### Considerações
- Adstock durante períodos de pico pode ter decays diferentes
- Competidores também aumentam investimento (considerar competitor_spend)
- Holiday_flag captura efeito geral, mas cenários específicos requerem ajuste

---

## Caso 5: Report Executivo Mensal

### Contexto
Gerar relatório mensal de performance de marketing para a diretoria.

### Conteúdo do Relatório

1. **Resumo executivo**: receita total, investimento total, ROAS geral
2. **Decomposição de receita**: contribuição de cada canal (waterfall chart)
3. **ROAS por canal**: ranking de eficiência
4. **Tendências**: evolução temporal das contribuições
5. **Recomendações**: canais para aumentar/reduzir investimento

### Exportação
- Dashboard interativo (Streamlit) para exploração
- PDF/Excel para apresentação executiva
- JSON via API para integração com sistemas internos

---

## Caso 6: Integração com Planejamento de Mídia

### Contexto
Integrar resultados do MMM com a plataforma de planejamento de mídia da agência.

### Fluxo de Integração

1. **API /train**: treina modelo com dados mais recentes
2. **API /coefficients**: fornece pesos de cada canal
3. **API /optimize**: recebe budget e retorna alocação ótima
4. **API /scenarios**: simula cenários propostos pela agência

### Automação
- Cron job semanal para retreinar modelo com novos dados
- Webhook para notificar equipe quando métricas mudam significativamente
- Dashboard atualizado em tempo real via Streamlit

---

## Caso 7: Análise Cross-Channel

### Contexto
Entender como a mídia offline (TV, rádio) amplifica a performance digital.

### Passos

1. Incluir features de interação (TV x Digital, TV x Search)
2. Analisar coeficientes de interação
3. Se positivos, existe efeito de amplificação cruzada
4. Quantificar o "halo effect" da TV sobre digital

### Interpretação
- Interação positiva: TV gera awareness que amplifica busca e conversão digital
- Interação negativa: canais competem pelo mesmo público (substituição)
- Sem interação significativa: canais operam de forma independente

---

*"A teoria sem prática é inútil, a prática sem teoria é perigosa." - Confúcio*
