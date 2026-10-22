"""Build the README and English report from recorded experiment results."""
from html import escape
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
TEAM = 'Barda Soltani Moakhar and Setayesh Heydari'
TITLE = 'Linear Algebra for Machine Learning'


def load(name):
    return json.loads((R/name).read_text())


def main():
    n,e,s,m = [load(x) for x in ['neural_metrics.json','embedding_metrics.json','sentiment_metrics.json','recommender_metrics.json']]
    g,c,rnn = n['gas_station'],n['classification'],n['rnn']
    improvement = 100*(m['bias_only_test_rmse']-m['factor_test_rmse'])/m['bias_only_test_rmse']
    readme = f'''# {TITLE}

Matrix methods for neural networks, word embeddings, and movie recommendations.

**Barda Soltani Moakhar · Setayesh Heydari**  
Linear Algebra course project | Instructor: **Maryam Ramezani**

This project connects three applications through the same tools: matrix multiplication, least squares, gradients, and low-rank representations. It starts with simple prediction models, builds a word-vector space from dialogue, and fits a sparse user-movie rating matrix. The main learning algorithms are implemented in NumPy, with explicit forward passes and gradients.

## Results at a glance

| Experiment | Evaluation | Recorded result |
| --- | --- | --- |
| Gas-station regression | Fourth queue, held out from fitting | R² = {g['validation_r2']:.4f}; slope = {g['slope_seconds_per_departure']:.2f} seconds per departure |
| Concentric-circle classification | 150 test examples | Logistic: {100*c['circles_logistic_test_accuracy']:.1f}%; ReLU network: {100*c['circles_mlp_test_accuracy']:.1f}% accuracy |
| Five-class softmax regression | 150 test examples | {100*c['blobs_softmax_test_accuracy']:.1f}% accuracy |
| Sentiment analysis | 1,000 held-out reviews | TF-IDF + logistic: {100*s['tfidf_test_accuracy']:.1f}%; PPMI + LSTM: {100*s['lstm_test_accuracy']:.1f}% accuracy |
| Movie rating prediction | 10,084 held-out ratings | Rank-20 factorization RMSE: {m['factor_test_rmse']:.4f}; bias-only baseline: {m['bias_only_test_rmse']:.4f} |

These are single-seed course experiments. The circle example illustrates a geometric difference between model classes. The sentiment result shows that a more complicated model can perform worse when the representation and training data are limited. Rating RMSE measures prediction error; it does not establish recommendation quality.

![Linear and nonlinear classifiers](results/figures/classification.png)

## 1. Neural networks and optimization

The first part fits a zero-intercept line to departure times from four gas-station queues. The first three queues supply 63 training observations; the fourth supplies 21 validation observations. Both least squares and gradient descent estimate about {g['slope_seconds_per_departure']:.2f} seconds per additional departure. This is elapsed time after the first observed departure, rather than a direct measurement of each driver's total wait.

Binary logistic regression, multiclass softmax regression, and a two-layer ReLU network then connect affine maps to classification. The circle experiment uses the same training and test observations for both classifiers. A character RNN implements truncated backpropagation through time and Adagrad on the included short demo corpus. Its generated samples illustrate what this small model learns and where it fails.

## 2. Word vectors and sentiment

The second part creates a sparse 3,000-by-3,000 word-context matrix from Simpsons dialogue. A symmetric context window covers two tokens on each side. The pipeline compares raw-count SVD embeddings with positive pointwise mutual information (PPMI) embeddings, using 16, 32, and 64 dimensions for nearest-neighbor queries.

Cosine similarity, vector arithmetic, group neighbors, and PCA expose the geometry of the learned space. A 12-question analogy diagnostic reports vocabulary coverage separately from accuracy. It is too small and too poorly covered to support a general claim about analogy quality.

A NumPy LSTM uses frozen 32-dimensional PPMI embeddings to classify the supplied IMDb subset. Normalized duplicate reviews are removed before a stratified 60/20/20 split. The LSTM reads the first 100 tokens, starts each review with a fresh hidden state, masks padding, and selects its checkpoint by validation loss. A TF-IDF logistic classifier provides a stronger baseline on the same test rows. The two models differ in both representation and classifier; the comparison is not an isolated test of architecture.

Optional GloVe code loads external `glove.6B.300d.txt` vectors, evaluates analogies, and produces PCA and t-SNE figures. Those vectors were not supplied, so GloVe scores are not included among the measured results.

## 3. Movie recommendations

The final part fits the observed entries of the MovieLens rating matrix:

$$
\\hat r_{{ui}} = \\mu + b_u + c_i + p_u^Tq_i.
$$

The model uses a fixed training mean, user and movie biases, 20 latent factors, and L2 regularization. A random 80/10/10 split separates fitting, epoch selection, and testing. Missing entries are excluded from the objective. The reported factorization is trained by SGD; it is not a direct SVD of a zero-filled rating matrix.

The test RMSE is {m['factor_test_rmse']:.4f}, compared with {m['global_mean_test_rmse']:.4f} for the global mean and {m['bias_only_test_rmse']:.4f} for a regularized bias-only model. The gain over the bias-only model is {improvement:.2f}% in RMSE. The saved recommendation example includes movie titles, preserves MovieLens IDs, and excludes films already rated by the selected user. It is an illustrative list, not a ranking evaluation.

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
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
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
'''
    (ROOT/'README.md').write_text(readme)

    sections = []
    def section(title, elements):
        sections.append((title,elements))
    def p(text): return ('p',text)
    def eq(text): return ('eq',text)
    def table(header,rows): return ('table',[header]+rows)
    def fig(path,caption): return ('fig',(path,caption))

    section('1. Project overview',[
        p(f'{TITLE} connects prediction, language representation, and movie recommendation through matrix operations. The project asks how far linear structure can take each task and what additional assumptions are needed when a linear model is insufficient.'),
        p('The work has three parts. The first derives and trains regression and classification models, then implements a character RNN. The second constructs a word-vector space with sparse co-occurrence matrices and truncated SVD, and applies the vectors to sentiment classification. The third predicts missing movie ratings through regularized user-item factorization.'),
        table(['Experiment','Evaluation result'],[
            ['Gas-station least squares',f'Validation R-squared {g["validation_r2"]:.4f}'],
            ['Circles: logistic / ReLU network',f'{100*c["circles_logistic_test_accuracy"]:.1f}% / {100*c["circles_mlp_test_accuracy"]:.1f}% test accuracy'],
            ['Five-class softmax',f'{100*c["blobs_softmax_test_accuracy"]:.1f}% test accuracy'],
            ['IMDb: TF-IDF / PPMI-LSTM',f'{100*s["tfidf_test_accuracy"]:.1f}% / {100*s["lstm_test_accuracy"]:.1f}% test accuracy'],
            ['MovieLens rank-20 factorization',f'{m["factor_test_rmse"]:.4f} test RMSE'],
        ]),
        p('The main models use NumPy with explicit derivatives. Scikit-learn supplies synthetic datasets, data splitting, metrics, dimensionality-reduction utilities, and the TF-IDF logistic baseline. SciPy supplies sparse matrices, numerical functions, and a sparse least-squares solver.'),
        p('Two results shape the interpretation. A nonlinear hidden layer solves the concentric-circle example that a linear boundary cannot represent. For sentiment, the TF-IDF baseline substantially outperforms the LSTM. Model complexity alone does not compensate for weak input coverage or limited training data.'),
    ])
    section('2. Data and evaluation protocol',[
        table(['Dataset','Preparation and split'],[
            ['Gas station', '21 departures in each of 4 queues; first 3 queues train, fourth validates.'],
            ['Synthetic classes','500 circles and 500 five-class blobs; 350 train / 150 test in each task.'],
            ['Character text',f'{rnn["characters"]:,} characters in a purpose-written demo corpus; contiguous 80/20 split.'],
            ['Simpsons dialogue',f'{e["source_rows"]:,} source rows; {e["retained_sentences"]:,} retained sentences after cleaning.'],
            ['IMDb subset',f'{s["source_rows"]:,} source reviews; {s["deduplicated_rows"]:,} after cleaning and deduplication; 60/20/20.'],
            ['MovieLens',f'{m["ratings"]:,} ratings; 80/10/10 split with {m["test"]:,} test entries.'],
        ]),
        p('All random splits and initializations use seed 42. Synthetic classification splits are stratified. The softmax feature scaler is fitted only on the training rows. IMDb reviews are normalized before duplicate removal; conflicting labels for the same normalized text are removed. Vocabulary statistics for TF-IDF come from training reviews alone. The PPMI embedding corpus is separate Simpsons dialogue and uses no IMDb labels.'),
        p('The sentiment validation split selects the epoch with the lowest cross-entropy. The rating validation split selects the epoch with the lowest RMSE. Each selected model is then evaluated on a separate test split. The simple classification demonstrations use fixed hyperparameters, with no model search on their test examples.'),
        p('The supplied IMDb CSV has no documented sampling procedure or official train/test identifiers. Its results describe this local subset and random split, not the standard Stanford IMDb benchmark. The MovieLens snapshot is a development dataset from September 2018, and random rating splits do not simulate deployment on future ratings.'),
        p('The repository includes the small numeric datasets and the demo corpus. Simpsons and IMDb CSVs can be imported locally from the course archive. Their upstream redistribution terms were not supplied. GloVe vectors are an optional external input; no pretrained GloVe result is reported in this run. SHA-256 fingerprints and package versions are stored in results/environment.json.'),
    ])
    section('3. Least squares and gradient descent',[
        p('For each queue, timestamps are converted to seconds and shifted by its first observed departure. The feature x is the number of departures since that first car, starting at zero. The target y is elapsed time. The zero-intercept model is y_hat = wx; it estimates a service-throughput trend, rather than an individual waiting-time measurement.'),
        eq(r'J(w)=\frac{1}{n}\|xw-y\|_2^2,\qquad w^*=\frac{x^Ty}{x^Tx}'),
        p(f'The fitted slope is {g["slope_seconds_per_departure"]:.6f} seconds per departure. Training R-squared is {g["training_r2"]:.6f}; held-out-line R-squared is {g["validation_r2"]:.6f}, with RMSE {g["validation_rmse_seconds"]:.2f} seconds. Both the analytic solution and 30 gradient steps agree to less than 1e-10 in the slope.'),
        eq(r'w_{t+1}=w_t-\eta\frac{2}{n}x^T(xw_t-y)'),
        p('The MSE Hessian is 2(x^T x)/n. Gradient descent therefore converges when 0 < eta < n/(x^T x), approximately 0.007317 for this design. The run uses eta = 0.005. A loss defined with an additional factor of one half would have a different numerical learning-rate bound.'),
        fig('regression.png','Figure 1. Departure-time observations and convergence of gradient descent.'),
        p('A perfect R-squared would require every held-out point to lie exactly on the fitted line. Unequal service durations, driver delays, and timestamp variation make that unlikely. The first validation point is zero by construction, and only one queue is held out, so this score is limited evidence about generalization to other stations or days.'),
    ])
    section('4. Linear and nonlinear classification',[
        p('Binary logistic regression maps an affine score through the sigmoid. Its mean cross-entropy gradient is X^T(p-y)/n for the weights and the mean residual for the bias. The ten-point illustrative dataset is fitted correctly, but its 100% training accuracy is not a generalization estimate.'),
        eq(r'p=\sigma(Xw+b),\qquad \sigma(z)=\frac{1}{1+e^{-z}}'),
        p(f'Five-class softmax regression uses a stable row-wise normalization of logits and one-hot targets. It trains for 1,500 full-batch epochs at learning rate 0.1 on standardized features. Its test accuracy is {100*c["blobs_softmax_test_accuracy"]:.2f}% on 150 synthetic examples.'),
        eq(r'P_{ic}=\frac{e^{Z_{ic}}}{\sum_j e^{Z_{ij}}},\qquad \nabla_W L=\frac{1}{n}X^T(P-Y)'),
        p(f'For the concentric-circle data, logistic regression reaches {100*c["circles_logistic_test_accuracy"]:.2f}% test accuracy. A network with 16 ReLU units reaches {100*c["circles_mlp_test_accuracy"]:.2f}% after 5,000 epochs at learning rate 0.1. Both see the same 350 training and 150 test observations.'),
        fig('classification.png','Figure 2. Test-set boundaries and the training losses of the two circle classifiers.'),
        p('The sigmoid equals one half on w^T x + b = 0, a straight line. Such a boundary cannot separate an inner ring from a surrounding outer ring. ReLU units combine several affine regions, which allows the network to represent a bounded class region. The result demonstrates expressiveness on this controlled task; it is not an estimate of performance on natural data.'),
    ])
    section('5. Character-level recurrent modeling',[
        p('The recurrent model predicts the next character from the current input and a hidden state. The included corpus is a short narrative about the project, rather than a full-book collection. Its sorted character vocabulary makes the index mapping deterministic.'),
        eq(r'h_t=\tanh(W_xx_t+W_hh_{t-1}+b_h),\qquad p_t=\mathrm{softmax}(W_yh_t+b_y)'),
        p(f'The model has {rnn["hidden_size"]} hidden units and {rnn["vocabulary"]} character types. Training uses 2,000 randomly chosen windows of length 32, mean next-character cross-entropy, Adagrad with learning rate 0.08, and elementwise gradient clipping to [-5, 5]. Every independent random window starts at a zero hidden state. This avoids carrying unrelated context between windows.'),
        fig('rnn_training.png','Figure 3. Per-window character loss and a moving average during RNN training.'),
        table(['Evaluation','Mean negative log-likelihood'],[
            ['Uniform distribution over the character vocabulary',f'{rnn["uniform_nll"]:.4f} nats/character'],
            ['Training passage',f'{rnn["training_nll"]:.4f} nats/character'],
            ['Contiguous held-out passage',f'{rnn["heldout_nll"]:.4f} nats/character'],
        ]),
        p('Evaluation resets the hidden state for each non-overlapping 32-character window and omits the final incomplete window. The vocabulary is fixed from the included corpus before splitting; only the vocabulary inventory, not held-out transitions, is visible during fitting. Samples use temperature 0.8 and sampling seed 123. They contain plausible fragments and malformed words. The training-versus-held-out gap indicates overfitting to a very small corpus.'),
        p('Backpropagation through time sums the output loss contributions and propagates hidden-state derivatives backward through the shared recurrent weights. A finite-difference check verifies every parameter of a small test network before clipping is applied.'),
    ])
    section('6. Sparse word embeddings',[
        p(f'The dialogue pipeline retains {e["retained_sentences"]:,} sentences. It removes HTML, lowercases alphabetic tokens, removes English stop words, and keeps sentences with at least three tokens. It applies frequency-adjusted bigram merging, with minimum pair count 20 and score threshold 10. It detects {e["detected_phrases"]} phrase types. It does not lemmatize words.'),
        p(f'The vocabulary contains the 3,000 most frequent tokens with at least ten occurrences. A context window of two tokens adds symmetric counts within each sentence. Tokens omitted from the vocabulary retain their original positions when the context distance is measured. The resulting matrix has {e["cooccurrence_nonzeros"]:,} nonzero entries.'),
        eq(r'\mathrm{PPMI}_{ij}=\max\left(\log\frac{C_{ij}\sum_{ab} C_{ab}}{(\sum_b C_{ib})(\sum_a C_{aj})},0\right)'),
        p('Unobserved entries remain zero. Truncated randomized SVD uses seven power iterations and seed 42. The raw-count representation uses U_k; the PPMI representation uses U_k times the square root of the retained singular values. Comparisons at 16, 32, and 64 dimensions are qualitative and change both matrix weighting and coordinate scaling. They do not isolate the effect of PPMI alone.'),
        fig('embeddings.png','Figure 4. Frequent tokens and the retained singular spectrum of the PPMI matrix.'),
        p(f'The 12-question analogy diagnostic covers only {e["analogy_evaluation"]["covered"]} questions and answers {e["analogy_evaluation"]["correct"]} correctly. Covered accuracy is {100*e["analogy_evaluation"]["accuracy_covered"]:.0f}%, but coverage is {100*e["analogy_evaluation"]["coverage"]:.1f}%. These counts are too small to characterize general analogy performance. Missing vocabulary is recorded rather than counted as a semantic error.'),
        p('The toy semantic-coordinate exercise has approximate values x = y = z = u = 0 and v = 1: fruit coordinates are small for people and high for oranges, while apples have no gender coordinate. Real learned axes are not generally interpretable as one named concept.'),
    ])
    section('7. Geometry and word-vector queries',[
        eq(r'\cos(u,v)=\frac{u^Tv}{\|u\|_2\|v\|_2}'),
        p('The vector-space implementation normalizes rows, excludes the query word from its own neighbors, and rejects zero-vector queries. An analogy a:b::c:? uses b-a+c, with all three input words excluded from the candidate set. Group queries average the member vectors and exclude group members. Least-similar vectors describe opposition in the learned geometry, not necessarily linguistic antonyms.'),
        fig('word_projection.png','Figure 5. PCA projection of selected PPMI vectors. Distances in this two-dimensional view do not preserve every relationship in the full space.'),
        table(['Query','Top PPMI neighbors (64 dimensions)'],[
            [w,', '.join(x[0] for x in e['vector_queries'].get('group_homer_bart',[]))] if w=='homer + bart' else
            [w,', '.join(pd.read_csv(R/'word_neighbors.csv').query('method == "PPMI SVD (U sqrt(S))" and dimensions == 64 and query == @w')['neighbor'].head(5))]
            for w in ['homer','lisa','bart','homer + bart']]),
        p('Dialogue embeddings reflect character names, repeated expressions, and the contexts present in one television corpus. A nearby word is an association learned from this text; it does not establish a personality trait. Small dimensionalities can emphasize frequent contexts and produce misleading neighbors.'),
        p('The optional GloVe experiment provides a separate loader, cosine queries, the same analogy diagnostic, PCA, and t-SNE. The recorded project did not include the external 300-dimensional vectors, so no GloVe score is claimed. A PCA or t-SNE picture is exploratory; neither is a quantitative semantic evaluation.'),
    ])
    section('8. Sentiment analysis with frozen embeddings',[
        p(f'The supplied review CSV contains 5,000 labeled rows. After normalized-text deduplication there are {s["deduplicated_rows"]:,} examples: {s["train"]:,} train, {s["validation"]:,} validation, and {s["test"]:,} test. The LSTM reads the first 100 tokens of each review and uses fixed, row-normalized 32-dimensional PPMI embeddings learned from dialogue.'),
        p(f'Unknown tokens and right-padding map to zero vectors. Padding positions preserve both cell and hidden states, with the same mask applied in the backward pass. Hidden and cell states start at zero for each review. Only {100*s["embedding_coverage_fraction"]:.1f}% of the retained review tokens occur in the dialogue vocabulary; common stop words, phrase mismatches, and domain-specific terms account for part of the missing coverage.'),
        eq(r'c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)'),
        p(f'The model has 24 hidden units and trains for eight epochs, with batch size 64, Adam learning rate 0.003, and global gradient-norm clipping at 5. Validation loss selects epoch {s["selected_epoch"]}. The TF-IDF baseline uses training-only unigram/bigram features, at most 10,000 features, minimum document frequency 2, sublinear term frequency, and logistic regularization C = 1.'),
        fig('sentiment.png','Figure 6. LSTM learning curves and accuracy on the same held-out review set.'),
        table(['Model','Test accuracy','Positive-class F1'],[
            ['Training-majority label',f'{100*s["majority_test_accuracy"]:.1f}%','-'],
            ['TF-IDF + logistic regression',f'{100*s["tfidf_test_accuracy"]:.1f}%',f'{s["tfidf_test_f1"]:.4f}'],
            ['Frozen PPMI + LSTM',f'{100*s["lstm_test_accuracy"]:.1f}%',f'{s["lstm_test_f1"]:.4f}'],
        ]),
        p(f'The baseline wins by {100*(s["tfidf_test_accuracy"]-s["lstm_test_accuracy"]):.1f} percentage points. Its review-domain features preserve useful sentiment phrases that may be absent from the dialogue vectors. The LSTM also sees only the first 100 tokens. These factors are plausible explanations, not isolated causal findings: this comparison changes the representation, sequence coverage, and classifier together.'),
    ])
    section('9. Matrix factorization for recommendation',[
        p(f'MovieLens contains {m["users"]} users, {m["rated_movies"]:,} movies with ratings, and {m["ratings"]:,} observed scores. Only {100*m["density"]:.2f}% of the user-movie matrix is observed. The catalog contains {m["catalog_movies"]:,} titles; some have tags but no ratings. Original IDs map to contiguous training indices and are restored in exported results.'),
        eq(r'\hat r_{ui}=\mu+b_u+c_i+p_u^Tq_i'),
        eq(r'L=\frac{1}{2}\sum_{(u,i)\in\Omega}\left[(r_{ui}-\hat r_{ui})^2+\lambda(b_u^2+c_i^2+\|p_u\|_2^2+\|q_i\|_2^2)\right]'),
        p(f'The objective includes observed training entries only. The global mean is fixed from training data. SGD uses rank 20, learning rate 0.01, regularization 0.05, shuffled observations, and 30 epochs. Validation selects epoch {m["selected_epoch"]}. The per-observation penalty means frequently observed users and items receive more total regularization.'),
        fig('recommendation.png','Figure 7. Factorization learning curves and test RMSE against two baselines.'),
        table(['Predictor','Test RMSE, no clipping'],[
            ['Global training mean',f'{m["global_mean_test_rmse"]:.4f}'],
            ['Regularized user and item biases',f'{m["bias_only_test_rmse"]:.4f}'],
            ['Rank-20 factors with biases',f'{m["factor_test_rmse"]:.4f}'],
        ]),
        p(f'The factor model reduces RMSE by {improvement:.2f}% relative to the bias-only baseline. That baseline is fitted by sparse least squares with ridge penalty 10 on each bias. The models use different fixed regularization conventions; this is a practical baseline comparison rather than a controlled ablation of the factor term. Clipping factor predictions to [0.5, 5] gives RMSE {m["factor_clipped_test_rmse"]:.4f}.'),
    ])
    section('10. Recommendation limits and reproducibility',[
        p(f'The test split contains {m["warm_start_test_ratings"]:,} warm-start ratings with RMSE {m["warm_start_test_rmse"]:.4f}, and {m["cold_start_test_ratings"]} ratings involving an item absent from training with RMSE {m["cold_start_test_rmse"]:.4f}. Unseen factor rows and biases remain zero, so their predictions fall back to the fixed mean and any learned counterpart bias. This split contains no unseen users.'),
        p('The illustrative top-ten list uses original userId 201 and movies with at least 20 training ratings. It removes every film the user rated in the supplied dataset, including held-out entries, before ranking candidates. That exclusion is for a known-history showcase and is not used to calculate rating RMSE. Reported display ratings are clipped, while ordering uses the raw score.'),
        p('A rating split can mix past and future behavior. RMSE also does not measure precision, recall, diversity, or user satisfaction for a recommendation list. A stronger follow-up would use a temporal split, explicit cold-start protocols, and ranking metrics. Multiple random seeds would help quantify variability in every learned model.'),
        p('Correctness checks compare every parameter gradient in small MLP, RNN, and masked LSTM instances with central finite differences. Additional checks cover stable softmax at extreme logits, the scalar least-squares solution, exact sparse context counts, nonnegative PPMI values, analogy input exclusion, padding invariance, and simultaneous factor updates.'),
        p('Reproduction uses scripts/run_experiments.py with --part neural, embeddings, sentiment, recommender, or all. Import the two external CSVs with scripts/prepare_data.py before the text experiments. Run python -m unittest discover -s tests -v for numerical checks. The notebooks read saved results unless RECOMPUTE is enabled. The README contains installation commands and the path to every deliverable.'),
        p('The measured run uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, and Matplotlib 3.10.8. Input hashes and exact versions are machine-readable. Small numerical differences can occur with other BLAS libraries or package versions; comparisons near a decision threshold may also change.'),
    ])
    section('Appendix A. Derivations and implementation details',[
        p('Least-squares complexity. For n rows and d features, forming X^T X costs O(nd^2) and a dense factorization/solve costs O(d^3). Forming X^T y adds O(nd). Thus a normal-equation implementation costs O(nd^2 + d^3), while T full-batch gradient steps cost O(Tnd). The scalar case is O(n) analytically and O(Tn) iteratively. The code uses lstsq, avoiding an explicit inverse; a QR-based solve also avoids squaring the condition number.'),
        p('Affine-layer composition. Removing the hidden activation gives Z_2 = (XW_1 + b_1)W_2 + b_2 = X(W_1W_2) + b_1W_2 + b_2. Define W_prime = W_1W_2 and b_prime = b_1W_2 + b_2. With a sigmoid output, this is a logistic model. For the binary setting, a nonempty hidden layer can represent any such affine score. In multiclass settings, a narrow bottleneck can impose an additional rank constraint.'),
        p('ReLU-network gradients. Let A_1 = ReLU(Z_1), P = sigmoid(Z_2), and D_2 = (P-y)/n. Then dW_2 = A_1^T D_2, db_2 = sum_rows(D_2), D_1 = (D_2 W_2^T) elementwise-multiplied by 1[Z_1 > 0], dW_1 = X^T D_1, and db_1 = sum_rows(D_1). Gradient checks avoid ReLU points at exactly zero, where the derivative is not unique.'),
        p('RNN gradients. For each timestep, dZ_y = (p_t - one_hot(y_t))/T. Add the recurrent contribution to W_y^T dZ_y, then multiply elementwise by (1-h_t^2) to differentiate tanh. Accumulate outer products into W_x, W_h, and W_y, and propagate W_h^T dZ_h to the previous step. Every timestep shares the same parameter matrices.'),
        p('LSTM gates. A single affine map of [x_t, h_(t-1)] supplies forget, input, output, and candidate gates. The first three use sigmoid and the candidate uses tanh. The forward and backward masks preserve states at right-padding positions. The classifier reads the final valid hidden state. Frozen embeddings receive no parameter updates.'),
        p('Factor gradients. With error e = r - r_hat, a user-factor step is p_u <- p_u + eta(e q_i - lambda p_u), and the item-factor step is q_i <- q_i + eta(e p_u_old - lambda q_i). Biases follow the same residual-minus-penalty rule. Copies of both factor vectors ensure that each update uses the same pre-update state.'),
        p('SVD and incomplete data. The truncated SVD gives a best rank-k Frobenius approximation to a fully observed matrix. The rating model instead optimizes a regularized loss over the observed index set Omega. It is nonconvex and is not guaranteed to reach that complete-matrix SVD solution. Filling missing ratings with zero changes the problem and introduces artificial low ratings.'),
    ])
    section('References and credits',[
        p(f'Project team: {TEAM}. Course: Linear Algebra. Instructor: Maryam Ramezani. The experiments develop the topics and datasets in Practical Assignment 2.'),
        p('F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. https://doi.org/10.1145/2827872'),
        p('Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. EMNLP. https://nlp.stanford.edu/projects/glove/'),
        p('Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781'),
        p('Andrew L. Maas and colleagues. 2011. Learning Word Vectors for Sentiment Analysis. ACL. Original dataset information: https://ai.stanford.edu/~amaas/data/sentiment/'),
        p('GroupLens. MovieLens latest-small dataset and usage conditions. https://grouplens.org/datasets/movielens/latest/ . The unchanged README accompanying the supplied September 2018 snapshot is included with the data.'),
        p('The Simpsons and IMDb CSVs are course-provided inputs. Their upstream versions were not documented in the archive. References to the original dataset projects do not establish an exact upstream match for these files. DATA_SOURCES.md records these limits and the local import procedure.'),
    ])

    markdown = [f'# {TITLE}\n\n{TEAM}  \nLinear Algebra | Instructor: Maryam Ramezani\n']
    for title,elements in sections:
        markdown.append('## '+title+'\n')
        for kind,value in elements:
            if kind=='p': markdown.append(value+'\n')
            elif kind=='eq': markdown.append('$$\n'+value+'\n$$\n')
            elif kind=='table':
                markdown.append('| '+' | '.join(value[0])+' |\n| '+' | '.join(['---']*len(value[0]))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in value[1:])+'\n')
            elif kind=='fig':
                filename,caption=value
                markdown.append(f'![{caption}](../results/figures/{filename})\n\n{caption}\n')
    (ROOT/'reports/report.md').write_text('\n'.join(markdown))
    make_pdf(sections)
    print(f'Built README.md, reports/report.md, and reports/report.pdf ({len(sections)} sections)')


def make_pdf(sections):
    from PIL import Image as PILImage
    from matplotlib import font_manager
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('ProjectSans',font_manager.findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('ProjectSansBold',font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))))
    styles = getSampleStyleSheet()
    body = ParagraphStyle('ProjectBody',parent=styles['BodyText'],fontName='ProjectSans',fontSize=9.2,leading=14,spaceAfter=9,textColor=colors.HexColor('#273746'))
    heading = ParagraphStyle('ProjectHeading',parent=styles['Heading1'],fontName='ProjectSansBold',fontSize=18,leading=23,textColor=colors.HexColor('#12364a'),spaceAfter=16)
    caption_style = ParagraphStyle('Caption',parent=body,fontSize=8,leading=10,textColor=colors.HexColor('#526875'),spaceAfter=12)
    table_style = ParagraphStyle('TableBody',parent=body,fontSize=8.5,leading=11,spaceAfter=0)
    story=[]
    page_width = 6.95*inch
    for index,(title,elements) in enumerate(sections):
        if index:
            story.append(PageBreak())
        if index==0:
            story.append(Paragraph(TITLE,ParagraphStyle('CoverTitle',parent=heading,fontSize=27,leading=31,spaceAfter=12)))
            story.append(Paragraph('NEURAL NETWORKS / WORD VECTORS / RECOMMENDATIONS',caption_style))
            story.append(Paragraph(TEAM,body))
            story.append(Paragraph('Linear Algebra course project<br/>Instructor: Maryam Ramezani',body))
            story.append(Spacer(1,14))
        story.append(Paragraph(escape(title),heading))
        for kind,value in elements:
            if kind=='p':
                story.append(Paragraph(escape(value),body))
            elif kind=='table':
                content=[[Paragraph(escape(str(x)),table_style) for x in row] for row in value]
                first = page_width*(.36 if len(value[0])==2 else .52)
                widths=[first]+[(page_width-first)/(len(value[0])-1)]*(len(value[0])-1)
                t=Table(content,colWidths=widths,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8faebd')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d4dfe4')),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
                story.extend([t,Spacer(1,12)])
            elif kind=='fig':
                filename,caption=value
                path=R/'figures'/filename
                with PILImage.open(path) as im:
                    width,height=im.size
                draw_width=min(page_width,6.6*inch)
                draw_height=draw_width*height/width
                if draw_height>2.65*inch:
                    draw_width*=2.65*inch/draw_height
                    draw_height=2.65*inch
                story.append(KeepTogether([Image(str(path),width=draw_width,height=draw_height),Spacer(1,5),Paragraph(escape(caption),caption_style)]))
            elif kind=='eq':
                buffer=io.BytesIO()
                figure=plt.figure(figsize=(8,.55))
                figure.text(.01,.45,'$'+value+'$',fontsize=12,va='center')
                figure.savefig(buffer,format='png',dpi=200,bbox_inches='tight',pad_inches=.04)
                plt.close(figure)
                buffer.seek(0)
                with PILImage.open(buffer) as im:
                    w,h=im.size
                buffer.seek(0)
                width=min(w/200*inch,page_width)
                story.extend([Image(buffer,width=width,height=width*h/w),Spacer(1,10)])
    doc=SimpleDocTemplate(str(ROOT/'reports/report.pdf'),pagesize=(8.27*inch,11.69*inch),rightMargin=.66*inch,leftMargin=.66*inch,topMargin=.7*inch,bottomMargin=.65*inch,title=TITLE,author=TEAM,subject='Linear Algebra course project | Maryam Ramezani')
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#c5d4dc'))
        canvas.line(.66*inch,.5*inch,7.61*inch,.5*inch)
        canvas.setFillColor(colors.HexColor('#526875'))
        canvas.setFont('ProjectSans',8)
        canvas.drawString(.66*inch,.32*inch,'LINEAR ALGEBRA FOR MACHINE LEARNING')
        canvas.drawRightString(7.61*inch,.32*inch,str(doc.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    main()
