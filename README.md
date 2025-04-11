# Beyond English: Assessing Memorization of Translated Texts in Large Language Models

## Overview

This repository contains the research conducted for the poster titled **"Beyond English: Assessing Memorization of Translated Texts in Large Language Models"** as part of the ERSP 2023-24 program. The goal of this research is to assess memorization of books in English, Spanish, Vietnamese, and Turkish in LLMs. 

## 🚀 Hypotheses & Research Questions

### 1. Translation Memorization
- **Hypothesis:** Large language models (LLMs) memorize the content of translated books.
- **Follow-up Question:** Do the models perform better in English if the original work is in Turkish, Spanish, and Vietnamese?

### 2. Cross-Lingual Memorization
- **Hypothesis:** LLMs can transfer their memorization across languages.
- **Follow-up Question:** Can LLMs memorize translations into languages not present in their pre-training dataset, and will their performance remain strong for out-of-distribution languages?

## 👩🏻‍💻 Contributors
| ![Alisha Srivastava](https://avatars.githubusercontent.com/alishasrivas?s=100) | ![Nhat Minh Le](https://avatars.githubusercontent.com/nhminhle?s=100) | ![Emir Korukluogu](https://avatars.githubusercontent.com/emirkaan5?s=100) |
|:---:|:---:|:---:|
| [Alisha Srivastava](https://github.com/alishasrivas) | [Nhat Minh Le](https://github.com/nhminhle) | [Emir Korukluogu](https://github.com/emirkaan5) |

### Special Thanks 🌟
- **Chau Minh Pham** - For guiding our research and being our research mentor.
- **Dr. Marzena Karpinska** - For guiding our research and for her invaluable expertise. 
- **Dr. Mohit Iyyer** - For guiding our research and being our research advisor.

## 🏗️ Dataset Construction

### Collect 25 books in English, Turkish, Vietnamese, and Spanish.
-- from Project Guttenberg and Online Sources 

### Process Books for passages >39 tokens & containing one Named Entity
- Extract excerpts from different languages while ensuring they contain full sentences and a single named entity.
- Clean text of metadata and align excerpts across four languages.
- Retain excerpts that pass length checks, contain only one named entity, and verify alignment.

## Experiments

### 1. Setup
- **Models Used:**
  - OpenAI API for GPT-4o
  - Claude API for Claude
  - Unity + vLLM for LLaMA models
  - Fireworks API for LLaMA-3.1-405B-instruct

### 2. Experiment Types
- **Experiment 0:** Direct Probing
  - Assessing accuracy based on exact and fuzzy matches.
- **Experiment 1:** Name Cloze Task
  - Input excerpts with masked names and evaluate exact matches.
- **Experiment 2:** Prefix Probing/Continuation Generation
  - Prompting models to continue provided sentences and evaluating performance metrics.

## Analyses

- Model capacity and quantization effects.
- Examination of quotes and named entities prevalence.
- Investigate how prefix token counts affect model performance.

## Contact

For any inquiries or discussions related to this research, please contact:

Alisha Srivastava at alishasrivas@umass.edu.
Nhat Minh Le at nhatminhle@umass.edu
Emir Korukluoglu at ekorukluoglu@umass.edu
