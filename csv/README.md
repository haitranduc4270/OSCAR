1. Download the OSCAR dataset from Kaggle: [oscar-dataset](https://www.kaggle.com/datasets/hitrnc/oscar-data).
2. Extract it so processed CSVs live under the repo root, for example:

```text
OSCAR/csv/processed/
  LUNG/
    5-fold/
      fold_1/ ...
      fold_5/ ...
  BRCA/
    ...
  COADREAD/
    ...
```

Pre-generated 5-fold splits are included under each cohort’s `5-fold/` directory. Point `data.pre_split_base` in the config at the matching folder (see below).
