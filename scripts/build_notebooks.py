"""Create the three project notebooks and execute their recorded-result cells.

Uses Python execution for the plain-code cells; no notebook server is required.
The notebook extra installs Jupyter for interactive use.
"""
import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {'cell_type':'markdown','id':uuid4().hex[:8],'metadata':{},'source':text.strip().splitlines(keepends=True)}


def code(text):
    return {'cell_type':'code','id':uuid4().hex[:8],'metadata':{},'execution_count':None,'outputs':[],'source':text.strip().splitlines(keepends=True)}


SETUP = '''from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "src" / "laml").exists():
    ROOT = ROOT.parent
if not (ROOT / "src" / "laml").exists():
    raise RuntimeError("Open this notebook from the repository root or notebooks directory")
sys.path.insert(0, str(ROOT / "src"))
RECOMPUTE = False

def read_result(name):
    return json.loads((ROOT / "results" / name).read_text())

def show_figure(name):
    from PIL import Image
    display(Image.open(ROOT / "results" / "figures" / name))
'''


def write_notebook(name,cells):
    context = {'__name__':'__main__'}
    number = 0
    os.chdir(ROOT)
    for cell in cells:
        if cell['cell_type']!='code':
            continue
        number += 1
        cell['execution_count'] = number
        outputs = cell['outputs']
        def display(value):
            data = {'text/plain':[str(value)]}
            if isinstance(value,PILImage.Image):
                buffer=io.BytesIO()
                value.save(buffer,format='PNG')
                data['image/png']=base64.b64encode(buffer.getvalue()).decode()
            elif hasattr(value,'to_html'):
                data['text/html']=[value.to_html()]
            elif hasattr(value,'savefig'):
                buffer=io.BytesIO()
                value.savefig(buffer,format='png',dpi=150,bbox_inches='tight')
                data['image/png']=base64.b64encode(buffer.getvalue()).decode()
            outputs.append({'output_type':'display_data','data':data,'metadata':{}})
        context['display']=display
        buffer=io.StringIO()
        with redirect_stdout(buffer):
            exec(compile(''.join(cell['source']),f'{name}:cell-{number}','exec'),context)
        if buffer.getvalue():
            outputs.append({'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(keepends=True)})
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.14'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'notebooks'/name
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    print(f'Executed {number} code cells: {name}')


def main():
    credits='**Barda Soltani Moakhar and Setayesh Heydari**  \nLinear Algebra | Instructor: **Maryam Ramezani**'
    cells=[
        markdown('# Neural networks and optimization\n\n'+credits+'\n\nWe connect least squares, affine classification, nonlinear feature maps, and recurrent state updates. The models and gradients are in `src/laml/models.py`. Recorded results use seed 42.'),
        code(SETUP),
        code('''if RECOMPUTE:
    from laml.experiments import run
    run("neural")
metrics = read_result("neural_metrics.json")
display(pd.DataFrame([metrics["gas_station"]]).T.rename(columns={0:"Value"}))'''),
        markdown(r'''## Least squares as a projection

For the zero-intercept model $y\approx xw$, minimizing $\|xw-y\|_2^2$ gives $w^*=(x^Ty)/(x^Tx)$. The residual is orthogonal to the feature vector. The first three queues train the model; the fourth checks the trend on a held-out queue.

The feature counts departures after the first observed car. The target is elapsed time after that departure, rather than an individual driver's total wait.'''),
        code('''from laml.models import least_squares
times = pd.read_csv(ROOT / "data/raw/gas_station.csv")
seconds = times.apply(lambda column: pd.to_timedelta(column).dt.total_seconds())
seconds -= seconds.iloc[0]
x = np.tile(np.arange(len(times)), 3)
y = seconds.iloc[:, :3].to_numpy().T.ravel()
w = least_squares(x, y)
print(f"Slope: {w:.6f} seconds per departure")
print(f"Residual inner product: {x @ (w*x-y):.3e}")
show_figure("regression.png")'''),
        markdown(r'''## Why the hidden activation matters

A binary logistic classifier has boundary $w^Tx+b=0$. It cannot separate an inner circle from an outer ring. A ReLU layer combines affine regions and can represent a bounded region.

Without the hidden activation, $(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2)$, so the model reduces to a single affine map before the final sigmoid.'''),
        code('''display(pd.DataFrame([metrics["classification"]]).T.rename(columns={0:"Value"}))
show_figure("classification.png")'''),
        markdown(r'''## A complete backward pass

For mean binary cross-entropy, the output-logit derivative is $(P-y)/n$. The hidden derivative multiplies the upstream derivative by the ReLU mask. The source below is the implementation used in training; numerical tests check it against central finite differences.'''),
        code('''import inspect
from laml.models import MLP
print(inspect.getsource(MLP.loss_grad))'''),
        markdown(r'''## Character RNN

The model updates $h_t=\tanh(W_xx_t+W_hh_{t-1}+b_h)$ and predicts the next character with a softmax. Backpropagation through time accumulates gradients for the same weights across 32 positions. Independent random windows start with zero state.

The short included corpus keeps the experiment self-contained. The held-out loss and generated sample show the limits of training a language model on only a few thousand characters.'''),
        code('''display(pd.DataFrame([metrics["rnn"]]).T.rename(columns={0:"Value"}))
show_figure("rnn_training.png")
print((ROOT / "results/generated_text.txt").read_text())'''),
        markdown('## Interpretation\n\nHigh circle accuracy reflects an easy synthetic geometric task. The character RNN learns local patterns but produces malformed words. The full report gives the learning-rate condition for scalar MSE and compares the computational cost of normal equations and gradient descent.'),
    ]
    write_notebook('01_neural_networks.ipynb',cells)
    cells=[
        markdown('# Word vectors and sentiment\n\n'+credits+'\n\nA sparse word-context matrix becomes a compact vector space through SVD. We inspect its geometry and test frozen word vectors as inputs to a sentiment classifier. Full numerical results are recorded even when the course CSVs are unavailable locally.'),
        code(SETUP),
        code('''if RECOMPUTE:
    from laml.experiments import run
    run("embeddings")
    run("sentiment")
embedding_metrics = read_result("embedding_metrics.json")
sentiment_metrics = read_result("sentiment_metrics.json")
display(pd.DataFrame([{k:embedding_metrics[k] for k in ["source_rows","retained_sentences","vocabulary_size","cooccurrence_nonzeros","ppmi_nonzeros","dimensions"]}]).T.rename(columns={0:"Value"}))'''),
        markdown(r'''## Exact counts on a small example

The context window is measured in token positions. Sentences remain separate. A pair contributes both directions, so the matrix is symmetric. The following calculation is independent of the external dataset.'''),
        code('''from laml.text import cooccurrence, ppmi
words, counts, _ = cooccurrence([["garden", "green", "trees"], ["garden", "green", "flowers"]], min_count=1, window=1)
display(pd.DataFrame(counts.toarray(),index=words,columns=words))
display(pd.DataFrame(ppmi(counts).toarray(),index=words,columns=words).round(3))'''),
        markdown(r'''## PPMI and SVD

We replace raw counts with $\max(\log(C_{ij}N/(C_{i\cdot}C_{\cdot j})),0)$ at observed entries. The raw-count baseline uses $U_k$; the PPMI space uses $U_k\Sigma_k^{1/2}$. Rows become word vectors. This comparison changes both weighting and coordinate scaling.

Cosine similarity compares directions. Nearest neighbors exclude the query token. The analogy $a:b::c:?$ uses $b-a+c$ and excludes all three input words.'''),
        code('''show_figure("embeddings.png")
show_figure("word_projection.png")
neighbors = pd.read_csv(ROOT / "results/word_neighbors.csv")
display(neighbors[(neighbors["dimensions"]==64) & (neighbors["query"]=="homer")])'''),
        code('''from laml.text import VectorSpace
toy = VectorSpace(["man", "woman", "king", "queen"], [[1,0],[0,1],[2,0],[1,1]])
print("Toy analogy man:woman::king:?", toy.analogy("man","woman","king"))
evaluation = embedding_metrics["analogy_evaluation"]
display(pd.DataFrame(evaluation["predictions"]))
print(f"Coverage: {evaluation['covered']}/{evaluation['total']}; correct: {evaluation['correct']}")
print("GloVe status:", embedding_metrics["glove_status"])'''),
        markdown('## A three-dimensional projection\n\nA third principal component can separate points that overlap in two dimensions. This view is still a projection and does not replace similarity measurements in the original vector space.'),
        code('''import matplotlib.pyplot as plt
projection = pd.read_csv(ROOT / "results/word_projection_3d.csv")
figure = plt.figure(figsize=(8, 5))
axis = figure.add_subplot(111, projection="3d")
axis.scatter(projection["pc1"],projection["pc2"],projection["pc3"],color="#157f83")
for row in projection.itertuples():
    axis.text(row.pc1,row.pc2,row.pc3,row.word,fontsize=8)
axis.set(xlabel="PC 1",ylabel="PC 2",zlabel="PC 3",title="PCA of selected PPMI vectors")
display(figure)
plt.close(figure)'''),
        markdown('## Frozen embeddings for sentiment\n\nThe LSTM receives 32-dimensional dialogue embeddings and the first 100 tokens of each review. Unknown tokens map to zero. Padding preserves the hidden and cell states, and reviews start with independent zero states. Validation selects the epoch; the test set is used for the final evaluation.\n\nThe TF-IDF baseline learns its vocabulary only from training reviews and uses unigram and bigram features. It sees full reviews. This comparison changes both representation and classifier.'),
        code('''display(pd.DataFrame([sentiment_metrics]).T.rename(columns={0:"Value"}))
predictions = pd.read_csv(ROOT / "results/sentiment_test_predictions.csv")
for column in ["tfidf_prediction", "lstm_prediction"]:
    accuracy = (predictions[column] == predictions["true_label"]).mean()
    print(f"{column}: {accuracy:.3f}")
show_figure("sentiment.png")'''),
        markdown('## Interpretation\n\nThe LSTM outperforms the majority label but trails TF-IDF substantially. Its low vocabulary coverage and truncated input are plausible contributors. The small analogy diagnostic has only two covered questions, so covered accuracy is not a useful general benchmark.\n\nOptional pretrained-vector analysis is available through `scripts/evaluate_glove.py`. It supports GloVe loading, analogies, PCA, and t-SNE. Those pretrained vectors are an external input and were not used in the recorded run.'),
    ]
    write_notebook('02_word_embeddings.ipynb',cells)
    cells=[
        markdown('# Sparse movie rating prediction\n\n'+credits+'\n\nWe learn low-rank user and movie factors from observed MovieLens ratings. The model includes global, user, and item biases. Missing entries do not enter the training loss.'),
        code(SETUP),
        code('''if RECOMPUTE:
    from laml.experiments import run
    run("recommender")
metrics = read_result("recommender_metrics.json")
display(pd.DataFrame([metrics]).T.rename(columns={0:"Value"}))'''),
        markdown(r'''## Observed-entry factorization

The predictor is $\hat r_{ui}=\mu+b_u+c_i+p_u^Tq_i$. SGD minimizes squared residuals plus a per-observation L2 penalty. Copies of both old factor vectors are used for simultaneous updates.

Ordinary truncated SVD solves a complete-matrix approximation problem. This regularized observed-entry objective is different. Filling unknown ratings with zero would add artificial low ratings.'''),
        code('''ratings = pd.read_csv(ROOT / "data/raw/movielens/ratings.csv")
print("Ratings:", len(ratings))
print("Users:", ratings["userId"].nunique())
print("Rated movies:", ratings["movieId"].nunique())
display(ratings.head())
show_figure("recommendation.png")'''),
        markdown('## Held-out evaluation\n\nA random 80/10/10 rating split separates training, validation, and test observations. Validation chooses the epoch. The bias-only baseline is fitted by a sparse regularized least-squares solve. Metrics below use raw, unclipped predictions.'),
        code('''predictions = pd.read_csv(ROOT / "results/rating_test_predictions.csv")
rmse = np.sqrt(np.mean((predictions["rating"]-predictions["prediction"])**2))
print(f"Recomputed test RMSE: {rmse:.6f}")
assert np.isclose(rmse, metrics["factor_test_rmse"])
display(pd.DataFrame({"Model":["Global mean","User + item biases","Rank-20 factors"], "Test RMSE":[metrics["global_mean_test_rmse"],metrics["bias_only_test_rmse"],metrics["factor_test_rmse"]]}))'''),
        markdown('## Recommendations with original identifiers\n\nThe showcase uses original MovieLens userId 201. It excludes every movie rated by that user in the supplied snapshot and limits candidates to films with at least 20 training ratings. Ordering uses the raw model score; the display rating is clipped to the valid range.'),
        code('''recommendations = pd.read_csv(ROOT / "results/recommendations_user_201.csv")
seen = set(ratings.loc[ratings["userId"]==201,"movieId"])
assert not (set(recommendations["movieId"]) & seen)
display(recommendations[["movieId","title","genres","predicted_rating"]])'''),
        markdown('## Limits\n\nRandom rating splits mix time periods. RMSE measures rating prediction rather than ranking quality. The popularity threshold and known-history exclusion are for the illustrative list; they are not a top-k benchmark. Cold-start results are reported separately. A temporal split and ranking metrics would be necessary to evaluate a deployed recommender.'),
    ]
    write_notebook('03_recommendation_system.ipynb',cells)


if __name__=='__main__':
    main()
