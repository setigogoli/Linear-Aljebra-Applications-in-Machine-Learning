"""Reproducible experiments; figures and numerical results share one source."""

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from sklearn.datasets import make_blobs, make_circles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as SklearnLogistic
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .models import CharacterRNN, LogisticRegression, LSTMClassifier, MLP, MatrixFactorization, SoftmaxRegression, least_squares
from .text import VectorSpace, cooccurrence, evaluate_analogies, phrase_tokens, ppmi, project_vectors, svd_embeddings, tokenize

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
SEED = 42
ANALOGIES = [
    ['paris','france','toronto','canada'], ['australia','dollar','japan','yen'],
    ['chicago','illinois','stockton','california'], ['brother','sister','father','mother'],
    ['apparent','apparently','rapid','rapidly'], ['great','greater','tough','tougher'],
    ['easy','easiest','lucky','luckiest'], ['think','thinking','read','reading'],
    ['switzerland','swiss','turkey','turkish'], ['walking','walked','swimming','swam'],
    ['mouse','mice','dollar','dollars'], ['work','works','speak','speaks'],
]


def setup():
    (RESULTS / 'figures').mkdir(parents=True, exist_ok=True)
    (RESULTS / 'models').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'savefig.dpi': 180, 'axes.titleweight': 'bold'})


def write_json(name, value):
    (RESULTS / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def savefig(name, figure):
    figure.tight_layout()
    figure.savefig(RESULTS / 'figures' / name, bbox_inches='tight')
    plt.close(figure)


def run_neural():
    setup()
    times = pd.read_csv(ROOT/'data/raw/gas_station.csv')
    seconds = times.apply(lambda col: pd.to_timedelta(col).dt.total_seconds())
    seconds -= seconds.iloc[0]
    x = np.tile(np.arange(len(times)), 3)
    y = seconds.iloc[:, :3].to_numpy().T.ravel()
    val_x, val_y = np.arange(len(times)), seconds.iloc[:, 3].to_numpy()
    slope = least_squares(x, y)
    w, losses = 0., []
    for _ in range(30):
        w -= .005 * 2 * np.mean(x * (w*x-y))
        losses.append(float(np.mean((w*x-y)**2)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for col in seconds:
        axes[0].scatter(val_x, seconds[col], label=col, s=16, alpha=.7)
    axes[0].plot(val_x, slope*val_x, color='#172b4d', label='Least squares')
    axes[0].set(xlabel='Departures since the first car', ylabel='Elapsed time (seconds)', title='Gas-station departure times')
    axes[0].legend(fontsize=7)
    axes[1].semilogy(range(1,31), losses, color='#157f83')
    axes[1].set(xlabel='Gradient step', ylabel='Training MSE', title='Gradient descent converges')
    savefig('regression.png', fig)

    small_x = np.array([[2,3],[1,3],[2,4],[1,6],[4,2],[4,5],[5,7],[3,6],[4,4],[2,6]], dtype=float)
    small_y = np.array([0]*5+[1]*5)
    small = LogisticRegression(learning_rate=.03, epochs=10000).fit(small_x, small_y)
    circles, labels = make_circles(n_samples=500, factor=.5, noise=.05, random_state=SEED)
    train, test = train_test_split(np.arange(500), test_size=.3, random_state=SEED, stratify=labels)
    logistic = LogisticRegression().fit(circles[train], labels[train])
    mlp = MLP().fit(circles[train], labels[train])
    blobs, classes = make_blobs(n_samples=500, centers=5, n_features=2, random_state=SEED)
    btrain, btest = train_test_split(np.arange(500), test_size=.3, random_state=SEED, stratify=classes)
    scaler = StandardScaler().fit(blobs[btrain])
    softmax_model = SoftmaxRegression().fit(scaler.transform(blobs[btrain]), classes[btrain])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    xx, yy = np.meshgrid(np.linspace(-1.3,1.3,180), np.linspace(-1.3,1.3,180))
    grid = np.c_[xx.ravel(), yy.ravel()]
    for model, ax, name in [(logistic, axes[0], 'Logistic regression'), (mlp, axes[1], 'ReLU network')]:
        ax.contourf(xx, yy, model.predict(grid).reshape(xx.shape), alpha=.2, cmap='coolwarm')
        ax.scatter(circles[test,0], circles[test,1], c=labels[test], cmap='coolwarm', s=12)
        ax.set(title=name, xlabel='Feature 1', ylabel='Feature 2')
    axes[2].plot(mlp.losses, color='#157f83', label='ReLU network')
    axes[2].plot(logistic.losses, color='#d87b32', label='Logistic regression')
    axes[2].set(title='Training cross-entropy', xlabel='Epoch', ylabel='Mean loss')
    axes[2].legend(fontsize=8)
    savefig('classification.png', fig)

    corpus = (ROOT/'data/raw/demo_corpus.txt').read_text()
    vocab = sorted(set(corpus))
    mapping = {ch:i for i,ch in enumerate(vocab)}
    ids = np.array([mapping[ch] for ch in corpus])
    split = int(.8*len(ids))
    rnn = CharacterRNN(len(vocab)).fit(ids[:split])
    def sequence_loss(part):
        return float(np.mean([rnn.loss_grad(part[s:s+32], part[s+1:s+33])[0] for s in range(0,len(part)-33,32)]))
    sample = ''.join(vocab[i] for i in rnn.sample(mapping['T']))
    (RESULTS/'generated_text.txt').write_text('Character RNN sample | demo corpus | temperature 0.8 | seed 123\n\nT'+sample+'\n')
    fig, ax = plt.subplots(figsize=(7, 2.7))
    rolling = pd.Series(rnn.losses).rolling(50).mean()
    ax.plot(rnn.losses, alpha=.18, color='#157f83')
    ax.plot(rolling, color='#157f83', label='50-step moving average')
    ax.axhline(np.log(len(vocab)), color='#d87b32', linestyle='--', label='Uniform character baseline')
    ax.set(xlabel='Update', ylabel='Nats per character', title='Character RNN training')
    ax.legend(fontsize=8)
    savefig('rnn_training.png', fig)
    metrics = {
        'seed':SEED, 'gas_station':{'training_observations':len(x),'validation_observations':len(val_x),
        'slope_seconds_per_departure':slope,'gradient_descent_slope':w,'gradient_descent_learning_rate':.005,
        'training_r2':float(r2_score(y,slope*x)),'validation_r2':float(r2_score(val_y,slope*val_x)),
        'validation_rmse_seconds':float(np.sqrt(np.mean((slope*val_x-val_y)**2)))},
        'classification': {'small_binary_training_accuracy':float(accuracy_score(small_y,small.predict(small_x))),
            'circles_logistic_test_accuracy':float(accuracy_score(labels[test],logistic.predict(circles[test]))),
            'circles_mlp_test_accuracy':float(accuracy_score(labels[test],mlp.predict(circles[test]))),
            'blobs_softmax_test_accuracy':float(accuracy_score(classes[btest],softmax_model.predict(scaler.transform(blobs[btest])))),
            'circles_train':len(train),'circles_test':len(test),'mlp_hidden_units':16,'mlp_epochs':5000},
        'rnn':{'corpus':'Project demo corpus','characters':len(ids),'vocabulary':len(vocab),'training_characters':split,
               'heldout_characters':len(ids)-split,'hidden_size':48,'sequence_length':32,'steps':2000,
               'training_nll':sequence_loss(ids[:split]),'heldout_nll':sequence_loss(ids[split:]),
               'uniform_nll':float(np.log(len(vocab)))}}
    write_json('neural_metrics.json',metrics)
    return metrics


def run_embeddings():
    setup()
    path = ROOT/'data/external/simpsons_dataset.csv'
    if not path.exists():
        raise FileNotFoundError('Import the course CSVs with scripts/prepare_data.py --archive PATH')
    data = pd.read_csv(path)
    texts = data['spoken_words'].dropna().drop_duplicates()
    sentences = [tokenize(t, True) for t in texts]
    sentences = [s for s in sentences if len(s)>=3]
    sentences, phrases = phrase_tokens(sentences)
    words, counts, frequencies = cooccurrence(sentences)
    raw_vectors, raw_s = svd_embeddings(counts, weighting='u')
    weighted = ppmi(counts)
    vectors, singular = svd_embeddings(weighted)
    np.savez_compressed(RESULTS/'models/embeddings.npz', words=np.array(words), vectors=vectors)
    rows, detail = [], {}
    for method, vec in [('Count SVD (U)',raw_vectors),('PPMI SVD (U sqrt(S))',vectors)]:
        for dim in [16,32,64]:
            space = VectorSpace(words,vec[:,:dim])
            detail[f'{method}, {dim} dimensions'] = {w:space.neighbors(w) for w in ['homer','lisa','bart'] if w in space.index}
            for word in ['homer','lisa','bart','school','family']:
                if word not in space.index:
                    continue
                for rank,(neighbor,score) in enumerate(space.neighbors(word),1):
                    rows.append({'method':method,'dimensions':dim,'query':word,'rank':rank,'neighbor':neighbor,'cosine':score})
    pd.DataFrame(rows).to_csv(RESULTS/'word_neighbors.csv',index=False)
    space = VectorSpace(words,vectors)
    analytics = {'least_similar_to_homer':space.neighbors('homer',least=True),
                 'group_homer_bart':space.group_neighbors(['homer','bart']),
                 'bart_minus_man_plus_woman':space.analogy('man','bart','woman',top_n=6),
                 'random_vector_neighbors':space.query(np.random.default_rng(SEED).normal(size=64))}
    evaluation = evaluate_analogies(space, ANALOGIES)
    fig, axes = plt.subplots(1,2,figsize=(10,3.7))
    top = frequencies.most_common(12)
    axes[0].barh([a for a,b in top][::-1],[b for a,b in top][::-1],color='#157f83')
    axes[0].set(xlabel='Token count',title='Frequent retained tokens')
    axes[1].plot(range(1,65),singular,color='#157f83',label='PPMI matrix')
    axes[1].set(xlabel='Component',ylabel='Singular value',title='PPMI singular spectrum')
    savefig('embeddings.png',fig)
    chosen = [w for w in ['homer','marge','bart','lisa','maggie','family','school','teacher','student','father','mother','son','daughter','house','work','money','love','friend'] if w in space.index]
    fig, ax = plt.subplots(figsize=(7,4))
    projected = project_vectors(np.array([vectors[space.index[w]] for w in chosen]))
    from sklearn.decomposition import PCA
    projected_3d = PCA(n_components=3,random_state=SEED).fit_transform(np.array([vectors[space.index[w]] for w in chosen]))
    pd.DataFrame({'word':chosen,'pc1':projected_3d[:,0],'pc2':projected_3d[:,1],'pc3':projected_3d[:,2]}).to_csv(RESULTS/'word_projection_3d.csv',index=False)
    ax.scatter(projected[:,0],projected[:,1],color='#157f83')
    for word, (a,b) in zip(chosen,projected):
        ax.annotate(word,(a,b),xytext=(4,3),textcoords='offset points',fontsize=8)
    ax.set(title='PCA of selected PPMI word vectors',xlabel='Principal component 1',ylabel='Principal component 2')
    savefig('word_projection.png',fig)
    metrics = {'seed':SEED,'source_rows':len(data),'unique_nonmissing_utterances':len(texts),'retained_sentences':len(sentences),
               'vocabulary_size':len(words),'context_window':2,'min_count':10,'detected_phrases':len(phrases),
               'cooccurrence_nonzeros':counts.nnz,'ppmi_nonzeros':weighted.nnz,'dimensions':64,
               'preprocessing':'HTML removal, lowercase alphabetic tokens, English stop words, minimum 3 tokens, scored bigram merging; no lemmatization',
               'analogy_evaluation':evaluation,'vector_queries':analytics,'glove_status':'not_run_external_vectors_not_supplied'}
    write_json('embedding_metrics.json',metrics)
    return metrics


def run_sentiment():
    setup()
    embedding_path = RESULTS/'models/embeddings.npz'
    if not embedding_path.exists():
        run_embeddings()
    saved = np.load(embedding_path)
    words, vectors = list(saved['words']), saved['vectors'][:,:32]
    vectors = vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    space = {w:i+1 for i,w in enumerate(words)}
    table = np.vstack([np.zeros((1,32)),vectors])
    data = pd.read_csv(ROOT/'data/external/imdb_reviews.csv').dropna(subset=['review','sentiment'])
    source_rows = len(data)
    data['normalized'] = data['review'].map(lambda s:' '.join(tokenize(s)))
    # Drop conflicting duplicate texts rather than allowing label ambiguity or split leakage.
    conflicts = data.groupby('normalized')['sentiment'].nunique()
    data = data[~data['normalized'].isin(conflicts[conflicts>1].index)].drop_duplicates('normalized').reset_index(drop=True)
    labels = data['sentiment'].map({'positive':1,'negative':0}).to_numpy()
    if np.isnan(labels.astype(float)).any():
        raise ValueError('Unexpected sentiment labels')
    indices = np.arange(len(data))
    train_val,test = train_test_split(indices,test_size=.2,stratify=labels,random_state=SEED)
    train,val = train_test_split(train_val,test_size=.25,stratify=labels[train_val],random_state=SEED)
    tfidf = TfidfVectorizer(max_features=10000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    train_tfidf = tfidf.fit_transform(data.loc[train,'normalized'])
    baseline = SklearnLogistic(C=1,max_iter=1000,random_state=SEED).fit(train_tfidf,labels[train])
    baseline_pred = baseline.predict(tfidf.transform(data.loc[test,'normalized']))
    sequences = [tokenize(t)[:100] for t in data['review']]
    lengths = np.array([max(1,len(s)) for s in sequences])
    ids = np.zeros((len(data),100),dtype=int)
    for j,sentence in enumerate(sequences):
        ids[j,:len(sentence)] = [space.get(w,0) for w in sentence]
    model = LSTMClassifier(32,hidden_dim=24)
    rng = np.random.default_rng(SEED)
    def evaluate(part):
        logits = np.concatenate([model.forward(table[ids[b]],lengths[b])[0] for b in np.array_split(part,max(1,int(np.ceil(len(part)/128))))])
        loss = float(np.mean(np.logaddexp(0,logits)-labels[part]*logits))
        return loss,(logits>=0).astype(int)
    history, best_loss, best_params, best_epoch = [],float('inf'),None,0
    for epoch in range(1,9):
        order = rng.permutation(train)
        total_loss = 0.
        for start in range(0,len(order),64):
            b = order[start:start+64]
            loss,grad = model.loss_grad(table[ids[b]],lengths[b],labels[b])
            model.update(grad)
            total_loss += loss*len(b)
        val_loss,val_pred = evaluate(val)
        history.append({'epoch':epoch,'training_loss':total_loss/len(train),'validation_loss':val_loss,
                        'validation_accuracy':float(accuracy_score(labels[val],val_pred))})
        print(f'LSTM epoch {epoch}/8, validation loss {val_loss:.4f}',flush=True)
        if val_loss < best_loss:
            best_loss,best_params,best_epoch = val_loss,deepcopy(model.params),epoch
    model.params = best_params
    test_loss,lstm_pred = evaluate(test)
    pd.DataFrame(history).to_csv(RESULTS/'sentiment_history.csv',index=False)
    pd.DataFrame({'row':test,'true_label':labels[test],'tfidf_prediction':baseline_pred,'lstm_prediction':lstm_pred}).to_csv(RESULTS/'sentiment_test_predictions.csv',index=False)
    fig,axes = plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(range(1,9),[h['training_loss'] for h in history],label='Training')
    axes[0].plot(range(1,9),[h['validation_loss'] for h in history],label='Validation')
    axes[0].set(xlabel='Epoch',ylabel='Binary cross-entropy',title='LSTM on frozen PPMI embeddings')
    axes[0].legend()
    accuracies = [max(labels[train].mean(),1-labels[train].mean()),accuracy_score(labels[test],baseline_pred),accuracy_score(labels[test],lstm_pred)]
    # Majority-class test accuracy must use the training-set majority label.
    accuracies[0] = float(np.mean(labels[test] == int(labels[train].mean()>=.5)))
    axes[1].bar(['Majority','TF-IDF + logistic','PPMI + LSTM'],accuracies,color=['#a9b5bd','#157f83','#d87b32'])
    axes[1].set(ylim=(0,1),ylabel='Test accuracy',title='Same held-out reviews')
    axes[1].tick_params(axis='x',labelsize=8)
    savefig('sentiment.png',fig)
    metrics = {'seed':SEED,'source_rows':source_rows,'deduplicated_rows':len(data),'train':len(train),'validation':len(val),'test':len(test),
               'split':'60/20/20 stratified random split after normalized-text deduplication',
               'max_tokens':100,'embedding_dimensions':32,'lstm_hidden_size':24,'epochs':8,'batch_size':64,'learning_rate':.003,
               'embedding_coverage_fraction':float(sum(w in space for s in sequences for w in s)/sum(map(len,sequences))),
               'selected_epoch':best_epoch,'majority_test_accuracy':accuracies[0],
               'tfidf_test_accuracy':float(accuracy_score(labels[test],baseline_pred)),'tfidf_test_f1':float(f1_score(labels[test],baseline_pred)),
               'lstm_test_accuracy':float(accuracy_score(labels[test],lstm_pred)),'lstm_test_f1':float(f1_score(labels[test],lstm_pred)),
               'lstm_test_loss':test_loss}
    write_json('sentiment_metrics.json',metrics)
    return metrics


def run_recommender():
    setup()
    data = pd.read_csv(ROOT/'data/raw/movielens/ratings.csv')
    movies = pd.read_csv(ROOT/'data/raw/movielens/movies.csv')
    user_ids,item_ids = sorted(data.userId.unique()),sorted(data.movieId.unique())
    user_map,item_map = {v:i for i,v in enumerate(user_ids)},{v:i for i,v in enumerate(item_ids)}
    u,i,r = data.userId.map(user_map).to_numpy(),data.movieId.map(item_map).to_numpy(),data.rating.to_numpy()
    train,holdout = train_test_split(np.arange(len(data)),test_size=.2,random_state=SEED)
    val,test = train_test_split(holdout,test_size=.5,random_state=SEED)
    n_users,n_items = len(user_ids),len(item_ids)
    mean = float(r[train].mean())
    # Bias-only baseline: solve a sparse regularized least-squares system.
    rows = np.tile(np.arange(len(train)),2)
    columns = np.concatenate([u[train],n_users+i[train]])
    design = sparse.csr_matrix((np.ones(2*len(train)),(rows,columns)),shape=(len(train),n_users+n_items))
    biases = lsqr(design,r[train]-mean,damp=np.sqrt(10),atol=1e-10,btol=1e-10)[0]
    bias_pred = mean+biases[u[test]]+biases[n_users+i[test]]
    model = MatrixFactorization(n_users,n_items)
    model.initialize(u[train],i[train],r[train])
    rng = np.random.default_rng(SEED)
    history,best,best_values,best_epoch = [],float('inf'),None,0
    rmse = lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    for epoch in range(1,31):
        model.epoch(u[train],i[train],r[train],rng,learning_rate=.01,regularization=.05)
        train_rmse = rmse(r[train],model.predict(u[train],i[train]))
        val_rmse = rmse(r[val],model.predict(u[val],i[val]))
        history.append({'epoch':epoch,'training_rmse':train_rmse,'validation_rmse':val_rmse})
        if val_rmse<best:
            best,best_values,best_epoch = val_rmse,deepcopy((model.p,model.q,model.bu,model.bi)),epoch
        if epoch%5==0:
            print(f'Matrix factorization epoch {epoch}/30, validation RMSE {val_rmse:.4f}',flush=True)
    model.p,model.q,model.bu,model.bi = best_values
    pred = model.predict(u[test],i[test])
    warm = np.isin(i[test],np.unique(i[train])) & np.isin(u[test],np.unique(u[train]))
    test_rows = data.iloc[test].copy()
    test_rows['prediction'] = pred
    test_rows['warm_start'] = warm
    test_rows.to_csv(RESULTS/'rating_test_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(RESULTS/'recommender_history.csv',index=False)
    selected_user = 201
    uid = user_map[selected_user]
    # Exclude all known rated films, including held-out entries, for the showcase list.
    seen = set(data.loc[data.userId==selected_user,'movieId'])
    item_counts = data.iloc[train].movieId.value_counts()
    candidates = [mid for mid in item_ids if mid not in seen and item_counts.get(mid,0)>=20]
    scores = model.predict(np.full(len(candidates),uid),np.array([item_map[mid] for mid in candidates]))
    recommendations = pd.DataFrame({'movieId':candidates,'predicted_rating':np.clip(scores,.5,5)})
    recommendations['raw_score'] = scores
    recommendations = recommendations.merge(movies,on='movieId').sort_values(['raw_score','movieId'],ascending=[False,True]).head(10)
    recommendations.to_csv(RESULTS/'recommendations_user_201.csv',index=False)
    fig,axes = plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(range(1,31),[h['training_rmse'] for h in history],label='Training')
    axes[0].plot(range(1,31),[h['validation_rmse'] for h in history],label='Validation')
    axes[0].set(xlabel='Epoch',ylabel='RMSE (rating points)',title='Observed-entry matrix factorization')
    axes[0].legend()
    values = [rmse(r[test],mean),rmse(r[test],bias_pred),rmse(r[test],pred)]
    axes[1].bar(['Global mean','User + item biases','Rank-20 factors'],values,color=['#a9b5bd','#157f83','#d87b32'])
    axes[1].set(ylabel='Test RMSE (rating points)',title='Unclipped predictions')
    axes[1].tick_params(axis='x',labelsize=8)
    savefig('recommendation.png',fig)
    metrics = {'seed':SEED,'ratings':len(data),'users':n_users,'rated_movies':n_items,'catalog_movies':len(movies),
               'density':len(data)/(n_users*n_items),'train':len(train),'validation':len(val),'test':len(test),
               'rank':20,'epochs':30,'learning_rate':.01,'regularization':.05,'selected_epoch':best_epoch,
               'validation_rmse':best,'global_mean_test_rmse':values[0],'bias_only_test_rmse':values[1],'factor_test_rmse':values[2],
               'factor_clipped_test_rmse':rmse(r[test],np.clip(pred,.5,5)),
               'warm_start_test_ratings':int(warm.sum()),'cold_start_test_ratings':int((~warm).sum()),
               'warm_start_test_rmse':rmse(r[test][warm],pred[warm]),
               'cold_start_test_rmse':rmse(r[test][~warm],pred[~warm]) if (~warm).any() else None,
               'split':'80/10/10 random rating split; validation selects the epoch; no temporal or ranking claim',
               'recommendation_userId':selected_user,'recommendation_minimum_training_ratings':20}
    write_json('recommender_metrics.json',metrics)
    return metrics


def provenance():
    inputs = sorted((ROOT/'data/raw').rglob('*.csv')) + sorted((ROOT/'data/external').glob('*.csv')) + [ROOT/'data/raw/demo_corpus.txt']
    packages = ['numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl']
    write_json('environment.json',{'python':platform.python_version(),'seed':SEED,
        'packages':{p:importlib.metadata.version(p) for p in packages},
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.exists()}})


def run(part='all'):
    with threadpool_limits(limits=1):
        tasks = {'neural':run_neural,'embeddings':run_embeddings,'sentiment':run_sentiment,'recommender':run_recommender}
        for name,task in tasks.items():
            if part in ('all',name):
                print(f'Running {name}...',flush=True)
                task()
        provenance()
