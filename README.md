# TFG de Lingüística

## Objectiu

Aquest treball estudia els patrons d'atenció humana (registrats amb eye-tracking) i els patrons d'atenció computacionals (obtinguts amb models de llenguatge) en la tasca de classificació de textos sexistes. Es compara l'alineament entre les anotacions humanes, les mètriques d'eye-tracking i les atribucions de models transformer finetunejats.

## Corpus

Es parteix del corpus  [MuSeD (*Multimodal Sexism Detection Dataset*)](https://github.com/lauradegrazia/MuSeD_sexism_detection_videos), que conté transcripcions de vídeos de TikTok i BitChute. S'han seleccionat 40 texts (20 sexistes i 20 no sexistes) seguint criteris de coherència entre modalitats, acord total entre anotadors i longitud raonable. 10 participants (5 homes i 5 dones) van llegir els texts mentre un eye-tracker Tobii Pro Spectrum (600 Hz) registrava els seus moviments oculars.

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
- Divergència de Jensen-Shannon per comparar distribucions d'atenció humana vs. anotacions.

---

## Estructura del repositori

```
eye_tracker_sexism/
├── README.md                          # Aquest fitxer
├── data/ (omés, contacteu-me si hi voleu accés)
├── src/
│   ├── process_tobii_raw_data.ipynb   # Notebook principal: processament dades + anàlisi complet
│   ├── anotations_analysis.ipynb      # Anàlisi de les anotacions dels participants
│   ├── create_corpus_from_mused.ipynb # Creació del corpus seleccionat a partir de MuSeD
│   ├── bert_baseline.ipynb            # Entrenament model BERT baseline
│   ├── stats.ipynb                    # Estadístiques addicionals
│   │
│   ├── utils/
│   │   ├── tobii.py                   # Processament de dades Tobii: fixacions, regressions, TFD/FFD/FC, AOI
│   │   ├── regressions.py             # Pipeline complet: z-scores, permutacions, FDR, hotspots, Fisher
│   │   ├── metrics.py                 # Mètriques de distribució: entropia creuada, KL, JS
│   │   ├── mused.py                   # Càrrega i processament del corpus MuSeD i spans
│   │   ├── data_correction.py         # Correccions manuals de dades
│   │   └── train.py                   # Funcions d'entrenament de models
│   │
│   ├── explain/                       # Mètodes d'interpretabilitat (LRP, gradients)
│   ├── models/                        # Definicions de models (MLM, seq classification)
│   ├── scripts/                       # Scripts auxiliars (compressió, conversió parquet)
│   ├── viz/                           # Visualització d'anotacions span
│   └── pdfs_anotacions/               # PDFs amb les anotacions span per text
│
├── latex/
│   ├── main.tex                       # Document principal del TFG
│   ├── preamble.tex                   # Configuració LaTeX
│   ├── capitols/
│   │   ├── intro.tex                  # Introducció
│   │   ├── marc.tex                   # Marc teòric
│   │   ├── metodologia.tex            # Metodologia
│   │   ├── resultats.tex              # Resultats
│   │   ├── discussio.tex              # Discussió
│   │   └── conclusions.tex            # conclusions
│   └── figs/                          # Figures
└── .gitignore
```

## Execució

El notebook principal és `src/process_tobii_raw_data.ipynb`. Per executar-lo:

```bash
cd src
uv run jupyter lab
```

Les dependències es gestionen amb `uv` (veure `src/pyproject.toml`). Les principals són: `pandas`, `numpy`, `scipy`, `scikit-learn`, `nltk`, `seaborn`, `matplotlib`.

@conklinEyeTrackingGuideApplied2018:

# Tobii pro lab

Interface sucks for text. You can only use a "Screen" project type and add the texts one by one manually if you want to automatically get charachters/words as AOI. The "Advanced screen" project lets you upload a csv with the experiments, but not for text.
So the alternatives to automate the experiments are 1) E-Prime, 2) Use images and map coordinates to words when postprocessing the data.

For now, I'm testing manually.

Before exporting, you have to use Analyze/AOI Tool to define the words AOI for each stimulus.
For qualitative analysis it is better to manually define AOIs. For example the region that contains sexism, as well as the pre-critical and spillover regions.

# Metrics

Lexical variables are the primary influence on early fixation times, while higher-level (contextual, sentence or discourse) variables are likely to show an influence later on, via re-reading and regressions and via increased overall fixation times.

## Early

Skipping rate. Word not fixated on first pass.

First fixation duration (word AOI) is equivalent to first pass reading time (gaze duration) for a multi-word AOI, since it can have multiple fixations.

## Intermediate (regressions)

A regression is going back to a previous word, "fixation on a previous ROI once the eye gaze has entered a later ROI". If we carefully define AOIs some software can export them. Otherwise we have to manually compute the regressions from token X to Y.

Regressions can be interesting in both directions: Out(previous text <- X), In(X <- later text).

## Late

Total reading time, number of fixations, re-reading time, second reading time...

## Possibles experiments (future work)

- Llegir en veu alta
- Llegir mentre s'escolta l'àudio

# Bibliography

@ikhwantriLookingDeepEyes2023 does an exhaustive study of eye tracking tasks vs models. They do sentiment analysis, relation classification and question answering, test different interpretability methods and compare with LSTM, CNN and vanilla transformers.

**Datasets**: They use two English datasets:

# References

- MQA-RC [@soodInterpretingAttentionModels2020] not only introduces a dataset, but also interprets the results.

- [@sood_interpreting_2020] has a nice example picture.

## ZuCo

- ZuCo [@hollenstein_zuco_2018] (newer version @hollenstein_zuco_2020)

Only 12 participants, 400 sentiment sentences, 300 QA sentences, 407 relation class. sentences.

They do a linguistic assessment of the participants.

### Experimental design

### ZuCo 2.0

@hollenstein_zuco_2020 has 18 participants

