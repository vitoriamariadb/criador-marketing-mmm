# Marketing Mix Modeling (MMM)

Ferramenta de modelagem de mix de marketing para análise de contribuição de canais,
otimização de budget e planejamento de cenários.

## Funcionalidades

- Ingestão de dados CSV/Excel
- Feature engineering com adstock e curvas de saturação
- Modelos de regressão (Ridge, Lasso, ElasticNet)
- Otimização de orçamento
- Planejamento de cenários
- Dashboard interativo com Streamlit
- Visualizações com Plotly
- API REST com Flask

## Estrutura

```
src/
    ingestion/      - Ingestão de dados
    features/       - Engenharia de features
    models/         - Modelos de regressão
    optimization/   - Otimização de budget
    visualization/  - Gráficos e dashboards
    api/            - API REST
tests/              - Testes automatizados
notebooks/          - Notebooks de exemplo
docs/               - Documentação
data/
    raw/            - Dados brutos
    processed/      - Dados processados
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
streamlit run src/visualization/dashboard.py
```

## Licença

MIT
