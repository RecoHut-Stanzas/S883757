Jobs recommendations
==============================

This repository contains the implementation of:

- several recommender system models appropriate
for large-scale jobs recommendations,
- a hyperparameter tuning,
- evaluation metrics.

Currently implemented models:

- ALS/WRMF: proposed in [Collaborative Filtering for Implicit Feedback Datasets](https://www.researchgate.net/publication/220765111_Collaborative_Filtering_for_Implicit_Feedback_Datasets);
  implementation based on the [Implicit](https://implicit.readthedocs.io/en/latest/als.html) implementation
- Prod2vec: proposed in [E-commerce in Your Inbox: Product Recommendations at Scale](https://www.researchgate.net/publication/304350592_E-commerce_in_Your_Inbox_Product_Recommendations_at_Scale);
  implementation based on [Gensim](https://github.com/RaRe-Technologies/gensim) Word2vec implementation
- RP3Beta proposed in [Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications](https://www.researchgate.net/publication/312430075_Updatable_Accurate_Diverse_and_Scalable_Recommendations_for_Interactive_Applications)
- SLIM: proposed in [SLIM: Sparse Linear Methods for Top-N Recommender Systems](https://www.researchgate.net/publication/220765374_SLIM_Sparse_Linear_Methods_for_Top-N_Recommender_Systems)
- LightFM: proposed in [Metadata Embeddings for User and Item Cold-start Recommendations](https://www.researchgate.net/publication/280589936_Metadata_Embeddings_for_User_and_Item_Cold-start_Recommendations);
  implementation based on the original [LightFM](https://github.com/lyst/lightfm) implementation
  
## Environment configuration

If you use conda, set-up conda environment with a kernel (working with anaconda3):

 ```bash
 make ckernel
 ```

If you use virtualenv, set-up virtual environment with a kernel:

 ```bash
 make vkernel
 ```

Then activate the environment:

 ```bash
source activate jobs-research
 ```

## Steps to reproduce the results

### Getting data

The input data file *interactions.csv* should be stored in the directory *data/raw/your-dataset-name*.
For example, *data/raw/jobs_published/interactions.csv*.
The file is expected to contain the following columns: *user, item, event, timestamp*.

To reproduce our results [download](https://www.kaggle.com/olxdatascience/olx-jobs-interactions) the
**olx-jobs dataset** from Kaggle.

### Running

Execute the command:

```bash
python run.py
 ```

The script will:

- split the input data,
- run the hyperparameter optimization for all models,
- train the models,
- generate the recommendations,
- evaluate the models. <br>

#### Details about each step

By default script executes all aforementioned steps, namely:

```bash
--steps '["prepare", "tune", "run", "evaluate"]'
 ```

##### Step *prepare*

This step:

- loads the raw interactions,
- splits the interactions into the *train_and_validation* and *test* sets,
- splits the *train_and_validation* set into *train* and *validation* sets,
- prepares *target_users* sets for whom recommendations are generated,
- saves all the prepared datasets.

Due to the large size of our dataset, we introduced additional parameters which enable us
to decrease the size of the *train* and *validation* sets used in the hyperparameter tuning:

```bash
--validation_target_users_size 30000
--validation_fraction_users 0.2
--validation_fraction_items 0.2
 ```

##### Step *tune*

This step performs Bayesian hyperparameter tuning on the *train* and *validation* sets.
<br>
For each model, the search space and the tuning parameters are defined in the *src/tuning/config.py* file.
The results of all iterations are stored.

##### Step *run*

This step, for each model:

- loads the best hyperparameters (if available),
- trains the model,
- generates and saves recommendations,
- saves efficiency metrics.

##### Step *evaluate*

This step, for each model:

- loads stored recommendations,
- evaluates them based on the implemented metrics,
- displays and stores the evaluation results.

### Notebooks

#### data

Notebooks to analyze the dataset structure and distribution.

#### models

Notebooks to demonstrate the usage of the particular models.

#### evaluation

Notebooks to better understand the results.
They utilize recommendations and metrics generated during the execution of the *run* script.

## Project structure
```
.
├── [ 40K]  images
│   └── [ 36K]  process_flow.svg
├── [1.0K]  LICENSE
├── [6.3K]  Makefile
├── [223K]  nbs
│   └── [219K]  P245068_OLX_Job_Recommendations_using_LightFM_SLIM_ALS_and_baseline_models.ipynb
├── [634K]  notebooks
│   ├── [ 92K]  data
│   │   ├── [ 85K]  analysis.ipynb
│   │   └── [2.8K]  get_subset.ipynb
│   ├── [433K]  evaluation
│   │   ├── [254K]  check_tuning_process.ipynb
│   │   ├── [ 26K]  display_stored_performance.ipynb
│   │   ├── [ 30K]  evaluate_on_users_with_at_least_N_test_interactions
│   │   │   ├── [ 16K]  evaluation.ipynb
│   │   │   ├── [4.3K]  prepare_recommendations.ipynb
│   │   │   └── [6.2K]  recommendations_overlap.ipynb
│   │   ├── [ 16K]  evaluation.ipynb
│   │   ├── [1.7K]  helpers.py
│   │   ├── [ 95K]  metrics_per_user
│   │   │   ├── [ 75K]  evaluation-per-number-of-interactions.ipynb
│   │   │   ├── [7.7K]  evaluation-per-user.ipynb
│   │   │   ├── [4.2K]  evaluator.py
│   │   │   ├── [ 352]  helpers.py
│   │   │   └── [4.3K]  metrics.py
│   │   └── [6.2K]  recommendations_overlap.ipynb
│   └── [106K]  models
│       ├── [ 12K]  als
│       │   └── [7.6K]  runbook.ipynb
│       ├── [ 15K]  lightfm
│       │   └── [ 11K]  runbook.ipynb
│       ├── [ 11K]  perfect
│       │   └── [7.0K]  runbook.ipynb
│       ├── [ 11K]  prod2vec
│       │   └── [6.8K]  runbook.ipynb
│       ├── [ 11K]  random
│       │   └── [6.8K]  runbook.ipynb
│       ├── [ 500]  README.md
│       ├── [ 20K]  rp3beta
│       │   ├── [7.2K]  runbook.ipynb
│       │   └── [9.0K]  runbook-tuning.ipynb
│       ├── [ 11K]  slim
│       │   └── [7.1K]  runbook.ipynb
│       └── [ 11K]  toppop
│           └── [6.8K]  runbook.ipynb
├── [4.3K]  README.md
├── [  75]  requirements-dev.in
├── [1.4K]  requirements-dev.txt
├── [ 140]  requirements.in
├── [2.0K]  requirements.txt
├── [ 11K]  run.py
├── [ 312]  setup.cfg
└── [ 99K]  src
    ├── [9.1K]  common
    │   ├── [2.4K]  config.py
    │   ├── [2.7K]  helpers.py
    │   └── [   0]  __init__.py
    ├── [9.3K]  data
    │   ├── [1.5K]  initializer.py
    │   └── [3.8K]  splitting.py
    ├── [ 16K]  evaluation
    │   ├── [4.5K]  evaluator.py
    │   ├── [   0]  __init__.py
    │   └── [7.6K]  metrics.py
    ├── [ 50K]  models
    │   ├── [5.0K]  als.py
    │   ├── [ 634]  base.py
    │   ├── [8.7K]  lightfm.py
    │   ├── [1.3K]  model_factory.py
    │   ├── [5.3K]  perfect.py
    │   ├── [8.2K]  prod2vec.py
    │   ├── [3.7K]  random.py
    │   ├── [4.5K]  rp3beta.py
    │   ├── [5.5K]  slim.py
    │   └── [3.7K]  toppop.py
    └── [ 10K]  tuning
        ├── [4.4K]  bayessian.py
        └── [1.6K]  config.py

 1.0M used in 22 directories, 56 files
```
