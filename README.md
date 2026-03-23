---
title: "Attention patterns of human annotators and language models when classifying sexist texts"
author: Pol Pastells
bibliography: eye_tracking.bib
output: pdf_document
toc: true
geometry: margin=3cm
---

# General

TFG en català.

# Experimental design

Total fixation duration can correlate both with difficulty and likeability -> Ask if the text was difficult.

0. Demographics questionaire: edat i sexe i prou
1. Read text
2. Press space
3. Annotate on excel. Fields: sexist (binary, scale?), difficult (binary, scale?)

Primer binari -> escala 0-2

@conklinEyeTrackingGuideApplied2018:

## Sentences

- How many sentences (total/per individual)? Limited time ~10 min.

## Control sentences

- How many control sentences?

20 textos, 4 de control

Control, preguntes sobre el text

- Can they be from the same dataset? If so, should they be read first and then explain the task for the rest?

Can all sentences be control sentences?

## Compiling this document

`pandoc README.md --citepro -V fontsize:11pt -o readme.pdf`

# Tobii pro lab

Interface sucks for text. You can only use a "Screen" project type and add the texts one by one manually if you want to automatically get charachters/words as AOI. The "Advanced screen" project lets you upload a csv with the experiments, but not for text.
So the alternatives to automate the experiments are 1) E-Prime, 2) Use images and map coordinates to words when postprocessing the data.

For now, I'm testing manually.

Before exporting, you have to use Analyze/AOI Tool to define the words AOI.
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

# Data analysis

The exported data contains columns `AOI size` (useless as a column, is constant) and `AOI hit` for each AOI.
For example, for a single word the column name looks like 'AOI size [Text - tu]', if there's repeated words they appear like 'AOI size [Text - tu].1'.

## Eye-gaze features

Which features?

- Fixation count (FC)
- First fixation duration (FFD)
- Total fixation duration (TFD)

## Notes

Stimulated recall: mirar les anotacions amb el participant per veure (preguntar) en què estaven pensant. Aturar mínim cada 30s.

Escala sexisme [0-2] + confiança

Paraula - primera passada

Recalibrar cada 15min

TODO:

- instruccions detallades
- qüestionari participants

## Possibles experiments

- Llegir en veu alta
- Llegir mentre s'escolta l'àudio

# Bibliography

@ikhwantriLookingDeepEyes2023 does an exhaustive study of eye tracking tasks vs models. They do sentiment analysis, relation classification and question answering, test different interpretability methods and compare with LSTM, CNN and vanilla transformers.

**Datasets**: They use two English datasets:

- MQA-RC [@soodInterpretingAttentionModels2020]

- ZuCo [@hollenstein_zuco_2018] (newer version @hollenstein_zuco_2020)

[@sood_interpreting_2020] has a nice example picture.

# References
