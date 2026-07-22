# TFG de Lingüística

## Objectiu

Aquest treball estudia els patrons d'atenció humana (registrats amb eye-tracking) i els patrons d'atenció computacionals (obtinguts amb models de llenguatge) en la tasca de classificació de textos sexistes. Es compara l'alineament entre les anotacions humanes, les mètriques d'eye-tracking i les atribucions de models transformer fine-tunejats.

## Corpus

Es parteix del corpus [MuSeD (*Multimodal Sexism Detection Dataset*)](https://github.com/lauradegrazia/MuSeD_sexism_detection_videos), que conté transcripcions de vídeos de TikTok i BitChute. S'han seleccionat 40 texts (20 sexistes i 20 no sexistes) seguint criteris de coherència entre modalitats, acord total entre anotadors i longitud raonable. 10 participants (5 homes i 5 dones) van llegir els texts mentre un eye-tracker Tobii Pro Spectrum (600 Hz) registrava els seus moviments oculars.

## Models

S'entrenen tres models de transformador en la tasca de classificació binària de textos sexistes:

| Model | Arquitectura | Paràmetres | Pesos |
|-------|-------------|-----------|-------|
| **MrBERT** | ModernBERT | 150M | BSC-LT/MrBERT-es |
| **BETO** | BERT | 110M | dccuchile/bert-base-spanish-wwm-cased |
| **mBERT** | BERT | 178M | bert-base-multilingual-cased |

Cada model s'entrena en dues condicions de dades:
- **Filtrades**: 124 texts d'entrenament (corpus filtrat per acord d'anotadors)
- **Completes**: 360 texts d'entrenament (corpus complet MuSeD, excepte test)

Per a les explicacions s'empren tres mètodes d'interpretabilitat (Captum):
- **Saliency**: gradient del output respecte a l'embedding d'entrada
- **Input×Gradient**: gradient multiplicat per l'entrada
- **Integrated Gradients**: integració del gradient al llarg d'un camí d'interpolació

## Mètriques d'eye-tracking

S'analitzen tres tipus de mètriques per token (àrea d'interès):

- **TFD** (*Total Fixation Duration*): suma de les durades de totes les fixacions en un token, normalitzada per text.
- **FFD** (*First Fixation Duration*): durada de la primera fixació en un token.
- **FC** (*Fixation Count*): nombre total de fixacions en un token.
- **Regressions**: moviments de la mirada cap enrere (tokens anteriors). Es classifiquen com a *within-line* o *between-line* i s'analitzen amb z-scores respecte a les baselines individuals.

## Anàlisi estadística

- Correlació de Spearman entre TFD/FFD/FC i les anotacions span.
- Prova de Mann-Whitney per comparar mètriques entre textos sexistes i no sexistes, i entre tokens dins i fora dels spans anotats.
- Z-scores de regressions amb correcció FDR de Benjamini-Hochberg.
- Test de permutació (10.000 permutacions) per validar la significació.
- Prova exacta de Fisher per avaluar el solapament entre tokens significatius i spans anotats.
- Divergència de Jensen-Shannon, entropia creuada i KL per comparar distribucions d'atenció (humana vs. spans vs. model).

---

## Estructura del repositori

```
eye_tracker_sexism/
├── README.md
├── data/                                # Dades (omés del repo, contacteu si hi voleu accés)
│   ├── chosen_data_full.csv             # 40 texts + anotacions span
│   ├── mused_all_clean.csv              # Corpus filtrat (~164 texts)
│   ├── 01_Text/text_with_mmlabel.csv    # Corpus complet (~400 texts)
│   └── tobii/                           # Dades eye-tracking
│       ├── all_parquets/                # 40 fitxers parquet (1 per participant × text)
│       └── general.tsv                  # Metadades participants
│
├── src/
│   ├── process_tobii_raw_data.ipynb     # Notebook principal: eye-tracking + anàlisi
│   ├── anotations_analysis.ipynb        # Anàlisi de les anotacions dels participants
│   ├── create_corpus_from_mused.ipynb   # Creació del corpus seleccionat
│   │
│   ├── train_models.py                  # Entrenament 3 models × 2 condicions
│   ├── explain_models.py                # Extracció d'atribucions (Saliency, InputXGrad, IG)
│   ├── compare_results.py               # Comparació: model vs spans vs eye-tracking
│   │
│   ├── utils/
│   │   ├── tobii.py                     # Fixacions, regressions, TFD/FFD/FC, AOI
│   │   ├── regressions.py               # Z-scores, permutacions, FDR, hotspots, Fisher
│   │   ├── metrics.py                   # Entropia creuada, KL, JS
│   │   ├── mused.py                     # Càrrega corpus MuSeD i spans
│   │   ├── train.py                     # Funcions d'entrenament HuggingFace
│   │   └── data_correction.py           # Correccions manuals de dades
│   │
│   ├── checkpoints/                     # 6 checkpoints entrenats (omés del repo)
│   │   ├── mrbert_filtered/             # MrBERT, dades filtrades
│   │   ├── mrbert_full/                 # MrBERT, dades completes
│   │   ├── beto_filtered/               # BETO, dades filtrades
│   │   ├── beto_full/                   # BETO, dades completes
│   │   ├── mbert_filtered/              # mBERT, dades filtrades
│   │   └── mbert_full/                  # mBERT, dades completes
│   │
│   ├── explanations/                    # Atribucions per checkpoint (CSV)
│   │   ├── mrbert_filtered.csv
│   │   ├── mrbert_full.csv
│   │   ├── beto_filtered.csv
│   │   ├── beto_full.csv
│   │   ├── mbert_filtered.csv
│   │   └── mbert_full.csv
│   │
│   ├── explain/                         # Mètodes d'interpretabilitat (Captum/LRP)
│   ├── models/                          # Definicions de models (antic)
│   ├── scripts/                         # Scripts auxiliars
│   ├── viz/                             # Visualització d'anotacions span
│   └── pdfs_anotacions/                 # PDFs amb les anotacions span per text
│
├── latex/
│   ├── main.tex                         # Document principal del TFG
│   ├── capitols/
│   │   ├── intro.tex                    # Introducció
│   │   ├── marcz.tex                    # Marc teòric
│   │   ├── metodologia.tex              # Metodologia
│   │   ├── resultats.tex                # Resultats
│   │   ├── discussio.tex                # Discussió
│   │   └── conclusions.tex              # Conclusions
│   ├── tables/
│   │   ├── model_comparison.csv         # Resultats numèrics (Spearman, JS, solapament)
│   │   ├── model_comparison.tex         # Taules LaTeX auto-generades
│   │   └── model_typology.csv           # Mètriques per tipologia d'etiqueta
│   └── figs/                            # Figures
└── .gitignore
```

## Execució

### Entrenament de models

```bash
cd src
uv run python train_models.py
```

Entrena 3 models × 2 condicions = 6 checkpoints a `checkpoints/`. MrBERT utilitza batch_size=8 (150M paràmetres), els altres batch_size=32.

### Extracció d'explicacions

```bash
cd src
uv run python explain_models.py
```

Extreu atribucions de token per a cadascun dels 40 texts × 6 checkpoints × 3 mètodes.

### Comparació amb eye-tracking i spans

```bash
cd src
uv run python compare_results.py
```

Compara les explicacions dels models amb les anotacions span i les mètriques d'eye-tracking. Genera:
- Taules de correlació de Spearman (model vs. TFD/FFD/FC)
- Solapament top-20% salient vs. spans anotats
- Mètriques de distribució: entropia creuada, KL, JS (model vs. spans, model vs. humà)
- Desglossament per tipologia d'etiqueta span

### Notebook principal

```bash
cd src
uv run jupyter lab process_tobii_raw_data.ipynb
```

Les dependències es gestionen amb `uv` (veure `src/pyproject.toml`). Les principals són: `pandas`, `numpy`, `scipy`, `scikit-learn`, `nltk`, `seaborn`, `matplotlib`, `captum`, `transformers`, `torch`.
