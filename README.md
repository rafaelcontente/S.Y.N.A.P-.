# Validação e Fiabilidade — como reproduzir

Esta pasta contém a corrida de validação completa que sustenta
`Relatorio_Validacao_Fiabilidade_SYNAP.pdf`.

## Reproduzir do zero

```bash
cd synap
python3 validation/run_validation.py   # corre os 8 módulos reais, guarda results.json + datasets
python3 validation/build_charts.py     # gera os gráficos (charts/*.png)
python3 validation/build_report.py     # compila o PDF final
```

## Ficheiros

- `run_validation.py` — constrói um dataset de referência com **processo gerador
  conhecido** (correlações verdadeiras fixadas antes de gerar uma linha), corre
  os 8 módulos reais sobre ele, e guarda todas as métricas em `results.json`.
- `build_charts.py` — gera os 11 gráficos (PNG, 200 DPI) a partir de `results.json`.
- `build_report.py` — compila o PDF profissional (`reportlab`) com narrativa,
  tabelas e todos os gráficos.
- `results.json` — todas as métricas numéricas desta corrida (fonte da verdade
  para o relatório — nenhum número no PDF é inventado ou hardcoded).
- `*.parquet`, `contract.pkl` — os datasets e o Contrato desta corrida específica.
- `m7_output/`, `m8_output/` — os relatórios do Compilador e do Auditor desta corrida.

## Porquê um processo gerador conhecido, e não apenas um dataset real qualquer

Comparar sintético contra uma amostra real arbitrária não distingue "parece-se
por coincidência" de "recuperou a estrutura real". Aqui, as correlações
verdadeiras são fixadas ANTES de qualquer linha existir — o mesmo dataset serve
de âncora (o que o sistema vê) e de verdade fundamental (o que o sistema teria
de recuperar, mas nunca lhe é dado diretamente). Ver Secção 1 do PDF para a
metodologia completa.
