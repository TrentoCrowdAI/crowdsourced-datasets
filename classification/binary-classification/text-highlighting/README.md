# text-highlighting

This folder contains the datasets used in the paper

Ramírez, J.; Baez, M.; Casati, F.; and Benatallah, B. 2019. **Understanding the impact of text highlighting in crowdsourcing tasks**. In Proceedings of the Seventh AAAI Conference on Human Computation and Crowdsourcing, HCOMP 2019.

In a nutshell, the datasets represent two kinds of tasks:

- classification tasks with highlighting support.
- highlighting tasks, where the workers highlight evidence.


## classification tasks

In this task, workers classified documents based on a given predicate. 


### classification tasks using crowdsourced highlights
Files:
- `classification_amazon-crowd-highlights.csv`
- `classification_oa-crowd-highlights.csv`
- `classification_tech-crowd-highlights.csv`
- `classification_tech-3x12-crowd-highlights.csv`
- `classification_tech-6x6-crowd-highlights.csv`

### classification tasks using ML-generated highlights
Files:
- `classification_amazon-ML-highlights.csv`
- `classification_oa-ML-highlights.csv`
- `classification_tech-ML-highlights.csv`


## highlighting tasks

In this task, workers highlighted excerpts from documents that are relevant to a given predicate, to support future classification tasks.

### crowdsourced highlights

File: `crowdsourced_highlights.csv`.

### ML-generated highlights

File: `ml_highlights.csv`.