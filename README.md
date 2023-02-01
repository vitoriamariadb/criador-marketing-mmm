# Marketing Mix Modeling (MMM)

Ferramenta de modelagem de mix de marketing para analise de contribuicao de canais,
otimizacao de budget e planejamento de cenarios.

## Funcionalidades

- Ingestao de dados CSV/Excel
- Feature engineering com adstock e curvas de saturacao
- Modelos de regressao (Ridge, Lasso, ElasticNet)
- Otimizacao de orcamento
- Planejamento de cenarios
- Dashboard interativo com Streamlit
- Visualizacoes com Plotly
- API REST com Flask

## Estrutura

```
src/
    ingestion/      - Ingestao de dados
    features/       - Engenharia de features
    models/         - Modelos de regressao
    optimization/   - Otimizacao de budget
    visualization/  - Graficos e dashboards
    api/            - API REST
tests/              - Testes automatizados
notebooks/          - Notebooks de exemplo
docs/               - Documentacao
data/
    raw/            - Dados brutos
    processed/      - Dados processados
```

## Instalacao

```bash
pip install -r requirements.txt
```

## Uso

```bash
streamlit run src/visualization/dashboard.py
```

## Licenca

MIT
