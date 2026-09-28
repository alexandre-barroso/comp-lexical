# Comparação lexical em fala lida

37 falantes brasileiros; 7.993 segmentos elegíveis. `data/` contém somente seus TextGrids, metadados necessários e MFCCs dos WAVs do Speech Accent Archive. Nenhum áudio é distribuído. Fonte das gravações: Speech Accent Archive (Weinberger, 2015).

Python 3.13; em ambiente virtual:

```sh
python -m pip install -r requirements.txt
python analysis.py verify
python analysis.py all
```

`verify` confere dados, contagens e testes; `all` também reproduz avaliações, sensibilidades, quatro tabelas e figura em `results/`. `reference/` registra os valores de comparação e a conferência desta versão.

Reextração opcional: coloque em `audio/` os WAVs com os nomes de `data/metadata.csv`, por exemplo `portuguese10.wav`:

```sh
python analysis.py extract --output results/reextracted
python analysis.py all --features results/reextracted/features
```

Lemas de contagem, com Lean 4.34.0 instalado:

```sh
python analysis.py formal
```

