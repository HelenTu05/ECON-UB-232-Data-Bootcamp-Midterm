# Data

## `yelp_restaurants_clean.csv` (committed)

The analysis-ready dataset, produced by the final cell of
[`notebooks/01_yelp_eda.ipynb`](../notebooks/01_yelp_eda.ipynb). It is committed
so that the Streamlit dashboard runs immediately after cloning, with no
downloads required.

## Raw Yelp Open Dataset (not committed)

To re-run the notebook from scratch you need the raw JSON files from the
[Yelp Open Dataset](https://www.yelp.com/dataset). They total several GB and are
excluded via `.gitignore`.

Required files:

```
data/yelp_dataset/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
└── yelp_academic_dataset_checkin.json
```

The notebook reads `../data/yelp_dataset/` by default. To point it somewhere
else, set an environment variable before launching Jupyter:

```bash
export YELP_DATA_DIR=/path/to/yelp_dataset   # Windows: set YELP_DATA_DIR=...
```

Accessing the raw dataset requires agreeing to Yelp's terms of use, which is why
it is not redistributed here.
