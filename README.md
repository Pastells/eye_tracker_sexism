---
title: "Attention patterns of human annotators and language models when classifying sexist texts"
author: Pol Pastells
bibliography: eye_tracking.bib
output: pdf_document
toc: true
geometry: margin=3cm
---

# General

- En quin idioma es fa el TFG? Anglès -> paper; però també pot està bé fer-lo en català.

# Experimental design

Total fixation duration can correlate both with difficulty and likeability -> Ask if the text was difficult.

0. Demographics questionaire: age, sex, field?
1. Read text
2. Press space
3. Annotate on excel. Fields: sexist (binary, scale?), difficult (binary, scale?)

## Sentences

- How many sentences (total/per individual)? Limited time ~10 min.

## Control sentences

- How many control sentences?
- Can they be from the same dataset? If so, should they be read first and then explain the task for the rest?

## Compiling this document

`pandoc README.md --citepro -V fontsize:11pt -o readme.pdf`

# Tobii pro lab

Interface sucks for text. You can only use a "Screen" project type and add the texts one by one manually if you want to automatically get charachters/words as AOI. The "Advanced screen" project lets you upload a csv with the experiments, but not for text.
So the alternatives to automate the experiments are 1) E-Prime, 2) Use images and map coordinates to words when postprocessing the data.

For now, I'm testing manually. Before exporting, you have to use Analyze/AOI Tool to define the words AOI.

# Data analysis

The exported data contains columns `AOI size` (useless as a column, is constant) and `AOI hit` for each AOI.
For example, for a single word the column name looks like 'AOI size [Text - tu]', if there's repeated words they appear like 'AOI size [Text - tu].1'.

## Eye-gaze features

Which features?

- Fixation count (FC)
- First fixation duration (FFD)
- Total fixation duration (TFD)

# Bibliography

@ikhwantri_looking_2023 does an exhaustive study of eye tracking tasks vs models. They do sentiment analysis, relation classification and question answering, test different interpretability methods and compare with LSTM, CNN and vanilla transformers.

**Datasets**: They use two English datasets:

- MQA-RC [@sood_interpreting_2020]

- ZuCo [@hollenstein_zuco_2018] (newer version @hollenstein_zuco_2020)

# References
