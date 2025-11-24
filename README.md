---
title: 'Attention patterns of human annotators and language models when classifying sexist texts'
author: Pol Pastells
output: pdf_document
bibliography: eye_tracking.bib
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

# Data analysis

## Eye-gaze features

Which features?
- Fixation count (FC)
- First fixation duration (FFD)
- Total fixation duration (TFD)

# Bibliography

@ikhwantri_looking_2023 does an exhaustive study of eye tracking tasks vs models. They do sentiment analysis, relation classification and question answering, test different interpretability methods and compare with LSTM, CNN and vanilla transformers.

**Datasets**: They use two English datasets:
- MQA-RC @sood_interpreting_2020

- ZuCo @hollenstein_zuco_2018 (newer version @hollenstein_zuco_2020)
