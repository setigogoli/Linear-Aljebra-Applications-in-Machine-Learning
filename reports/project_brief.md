# Linear Algebra for Machine Learning

**Team:** Barda Soltani Moakhar and Setayesh Heydari  
**Course:** Linear Algebra  
**Instructor:** Maryam Ramezani

## Part 1: Learning with vectors and matrices

Fit a zero-intercept least-squares model to gas-station departure times. Derive its solution, implement gradient descent, and compare the fitted parameters. Train binary and multiclass linear classifiers, then compare logistic regression with a ReLU network on concentric circles. Implement the forward and backward passes of a character RNN and inspect its generated samples.

Required evidence: fitted slope, held-out-line R-squared, classification accuracy on fixed test splits, training curves, character-level cross-entropy, and numerical gradient checks. Explain why an affine decision boundary cannot separate concentric classes and why composing affine layers does not add nonlinear expressive power.

## Part 2: Language as a vector space

Build a sparse word-context matrix from the supplied Simpsons dialogue CSV. Compare embeddings from raw counts with embeddings from a positive pointwise mutual information matrix. Use truncated SVD, cosine similarity, vector arithmetic, nearest-neighbor queries, and PCA to inspect the representations.

Evaluate the word vectors as frozen inputs to an LSTM sentiment classifier on the supplied IMDb review subset. Compare the LSTM with a TF-IDF logistic baseline on the same held-out reviews. Keep repeated reviews within a single split by deduplicating before splitting.

An optional GloVe experiment loads external pretrained vectors, evaluates a small analogy set, and creates PCA and t-SNE plots. Record coverage and accuracy separately. Report the experiment as unrun when the vectors are unavailable.

## Part 3: Completing a sparse rating matrix

Train a regularized user-item factorization model on observed MovieLens ratings. Preserve original identifiers, use a validation split to select the epoch, and evaluate once on a separate test split. Compare with global-mean and user-plus-item-bias baselines. Produce a recommendation list with movie titles and exclude previously rated films.

Explain why missing ratings must not be treated as zeros and why fitting observed entries by SGD differs from applying an ordinary SVD to a fully observed matrix.

