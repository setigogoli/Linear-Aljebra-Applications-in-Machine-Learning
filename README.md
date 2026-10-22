# Linear Algebra for Machine Learning

Matrix methods for neural networks, word embeddings, and movie recommendations.

**Barda Soltani Moakhar · Setayesh Heydari**  
Linear Algebra course project | Instructor: **Maryam Ramezani**

This project connects three applications through the same tools: matrix multiplication, least squares, gradients, and low-rank representations. It starts with simple prediction models, builds a word-vector space from dialogue, and fits a sparse user-movie rating matrix. The main learning algorithms are implemented in NumPy, with explicit forward passes and gradients.

## Results at a glance

| Experiment | Evaluation | Recorded result |
| --- | --- | --- |
| Gas-station regression | Fourth queue, held out from fitting | R² = 0.9829; slope = 92.72 seconds per departure |
| Concentric-circle classification | 150 test examples | Logistic: 50.7%; ReLU network: 100.0% accuracy |
| Five-class softmax regression | 150 test examples | 96.7% accuracy |
| Sentiment analysis | 1,000 held-out reviews | TF-IDF + logistic: 87.4%; PPMI + LSTM: 65.5% accuracy |
| Movie rating prediction | 10,084 held-out ratings | Rank-20 factorization RMSE: 0.8791; bias-only baseline: 0.8896 |

These are single-seed course experiments. The circle example illustrates a geometric difference between model classes. The sentiment result shows that a more complicated model can perform worse when the representation and training data are limited. Rating RMSE measures prediction error; it does not establish recommendation quality.

![Linear and nonlinear classifiers](results/figures/classification.png)

## 1. Neural networks and optimization

The first part fits a zero-intercept line to departure times from four gas-station queues. The first three queues supply 63 training observations; the fourth supplies 21 validation observations. Both least squares and gradient descent estimate about 92.72 seconds per additional departure. This is elapsed time after the first observed departure, rather than a direct measurement of each driver's total wait.

Binary logistic regression, multiclass softmax regression, and a two-layer ReLU network then connect affine maps to classification. The circle experiment uses the same training and test observations for both classifiers. A character RNN implements truncated backpropagation through time and Adagrad on the included short demo corpus. Its generated samples illustrate what this small model learns and where it fails.

## 2. Word vectors and sentiment

The second part creates a sparse 3,000-by-3,000 word-context matrix from Simpsons dialogue. A symmetric context window covers two tokens on each side. The pipeline compares raw-count SVD embeddings with positive pointwise mutual information (PPMI) embeddings, using 16, 32, and 64 dimensions for nearest-neighbor queries.

Cosine similarity, vector arithmetic, group neighbors, and PCA expose the geometry of the learned space. A 12-question analogy diagnostic reports vocabulary coverage separately from accuracy. It is too small and too poorly covered to support a general claim about analogy quality.

A NumPy LSTM uses frozen 32-dimensional PPMI embeddings to classify the supplied IMDb subset. Normalized duplicate reviews are removed before a stratified 60/20/20 split. The LSTM reads the first 100 tokens, starts each review with a fresh hidden state, masks padding, and selects its checkpoint by validation loss. A TF-IDF logistic classifier provides a stronger baseline on the same test rows. The two models differ in both representation and classifier; the comparison is not an isolated test of architecture.

Optional GloVe code loads external `glove.6B.300d.txt` vectors, evaluates analogies, and produces PCA and t-SNE figures. Those vectors were not supplied, so GloVe scores are not included among the measured results.

## 3. Movie recommendations

The final part fits the observed entries of the MovieLens rating matrix:

$$
\hat r_{ui} = \mu + b_u + c_i + p_u^Tq_i.
$$

The model uses a fixed training mean, user and movie biases, 20 latent factors, and L2 regularization. A random 80/10/10 split separates fitting, epoch selection, and testing. Missing entries are excluded from the objective. The reported factorization is trained by SGD; it is not a direct SVD of a zero-filled rating matrix.

The test RMSE is 0.8791, compared with 1.0545 for the global mean and 0.8896 for a regularized bias-only model. The gain over the bias-only model is 1.17% in RMSE. The saved recommendation example includes movie titles, preserves MovieLens IDs, and excludes films already rated by the selected user. It is an illustrative list, not a ranking evaluation.

![Rating prediction results](results/figures/recommendation.png)

## Repository guide

| Path | Contents |
| --- | --- |
| `notebooks/01_neural_networks.ipynb` | Regression, linear and nonlinear classification, and character RNN |
| `notebooks/02_word_embeddings.ipynb` | Sparse embeddings, vector queries, projections, and sentiment results |
| `notebooks/03_recommendation_system.ipynb` | Factorization, baselines, test predictions, and recommendations |
| `src/laml/models.py` | NumPy models and explicit backward passes |
| `src/laml/text.py` | Tokenization, phrase detection, sparse matrices, SVD, and vector queries |
| `src/laml/experiments.py` | Fixed data splits, training, evaluation, and figure generation |
| `reports/project_brief.md` | Project questions and expected evidence |
| `reports/report.pdf` | English technical report |
| `reports/report.md` | Editable report source |
| `results/` | Recorded metrics, histories, predictions, and figures |
| `data/raw/` | Gas-station CSV, MovieLens snapshot, and demo text |
| `tests/` | Numerical gradient and data-handling checks |
| `DATA_SOURCES.md` | Data provenance, preparation, attribution, and reuse conditions |

## Run the project

Use Python 3.10 or later. The recorded run used Python 3.12; exact package versions and input hashes are in [`results/environment.json`](results/environment.json).

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

The regression, classification, character RNN, and MovieLens experiments run with the included data:

```bash
python scripts/run_experiments.py --part neural
python scripts/run_experiments.py --part recommender
python -m unittest discover -s tests -v
```

To reproduce the text experiments, import the two supplied course CSVs from the original assignment archive:

```bash
python scripts/prepare_data.py --archive "/path/to/PR_2 (2).zip"
python scripts/run_experiments.py --part embeddings
python scripts/run_experiments.py --part sentiment
```

With those CSVs prepared, run every experiment in order:

```bash
python scripts/run_experiments.py --part all
```

The text CSVs are kept outside version control because their upstream redistribution terms were not supplied. The full-book corpus from the assignment is also excluded. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for the exact files, columns, and source information. The report and recorded notebook outputs can be read without importing external data.

To open the notebooks or rebuild the report:

```bash
python -m pip install -e ".[notebooks,report]"
jupyter lab
# After rerunning experiments:
python scripts/build_docs.py
```

Each notebook reads the recorded results by default. Set `RECOMPUTE = True` in its setup cell to train its experiments again. That setting overwrites the corresponding files under `results/`. Run the scripts from a source checkout so the repository's data and result paths are available.

For the optional pretrained-vector experiment:

```bash
python scripts/evaluate_glove.py --file "/path/to/glove.6B.300d.txt"
```

Download the vector file from the [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/). This optional experiment can use substantially more memory than the included examples. `--limit` loads a vocabulary prefix, which changes analogy coverage and the set of candidate answers.

## Reproducibility and limitations

All experiments use seed 42. The report distinguishes training, validation, and test results. The RNN, MLP, and LSTM gradients are checked against central finite differences. Additional tests cover padding invariance, stable softmax, context-window counts, analogy exclusions, and factorization updates. GitHub Actions runs these checks on pushes and pull requests.

The report describes the evaluation limits: one seed, synthetic classification examples, a short RNN corpus, domain mismatch in the sentiment embeddings, and a random rating split rather than a future-rating simulation. The test sets were not used to select training epochs. Public result files include predictions and split membership so the reported scores can be inspected.

## Credits and references

Prepared by **Barda Soltani Moakhar** and **Setayesh Heydari** for the Linear Algebra course taught by **Maryam Ramezani**. The project is based on the topics and datasets in Practical Assignment 2.

MovieLens is provided by [GroupLens](https://grouplens.org/datasets/movielens/latest/). Its original usage conditions are preserved in `data/raw/movielens/README.txt`. See [`CREDITS.md`](CREDITS.md), [`DATA_SOURCES.md`](DATA_SOURCES.md), and the report for full references.
