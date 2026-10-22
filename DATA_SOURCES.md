# Data sources and reuse

| Data | Project use | Availability |
| --- | --- | --- |
| Gas-station departure times, 25 October 2024 | Zero-intercept least squares | Course-provided CSV in `data/raw/gas_station.csv`; 21 observations in each of four lines. |
| MovieLens `ml-latest-small`, generated 26 September 2018 | Rating prediction and recommendation | Supplied snapshot in `data/raw/movielens/`, including the unchanged GroupLens usage conditions in `README.txt`. |
| Simpsons dialogue CSV | Count and PPMI word embeddings | Course-provided `simpsons_dataset.csv`; imported locally into `data/external/`. An upstream version and redistribution grant were not included with the course archive. |
| IMDb review subset | Binary sentiment analysis | Course-provided `IMDB Dataset_small.csv`; imported locally as `data/external/imdb_reviews.csv`. It contains 5,000 rows. Its sampling procedure and relationship to the original Stanford train/test split are not documented. |
| Demo corpus | Character-level language modeling | Short, purpose-written narrative in `data/raw/demo_corpus.txt`. This is an educational corpus, not a language benchmark. |
| GloVe 6B, 300 dimensions | Optional pretrained-vector analysis | Download separately from the [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/). Use `glove.6B.300d.txt`, not the newer 2024 vector set, to match this experiment. |

The original archive also contains a preprocessed collection of the Harry Potter books. That collection is not distributed in this repository. The character RNN runs on the included demo corpus.

## Preparing the local text data

From the repository root, run:

```bash
python scripts/prepare_data.py --archive "/path/to/PR_2 (2).zip"
```

The script copies only the two text CSVs, verifies that each filename resolves to one archive entry, and records SHA-256 hashes. It does not require a network connection or extract the archive to arbitrary paths. `data/external/` is ignored by Git.

The public repository includes measured outputs and notebooks for the text experiments. Repeating those experiments requires the original course CSVs or replacement CSVs with the same columns. Substituting other data changes the reported results.

## MovieLens attribution and conditions

MovieLens is provided by GroupLens at the University of Minnesota. Its supplied usage conditions permit redistribution under the same conditions and require permission for commercial or revenue-bearing use. Those conditions continue to apply to the data and its transformations. The repository does not imply endorsement by GroupLens or the University of Minnesota.

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. https://doi.org/10.1145/2827872

Dataset page: https://grouplens.org/datasets/movielens/latest/

The copied 2018 snapshot is identified by file hashes in `results/environment.json`. Downloading the latest archive later may yield different bytes. The project is a course experiment; it does not present this development dataset as a fixed research benchmark.

## Other references

The [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is the original source commonly associated with IMDb sentiment experiments. Its published benchmark scores and predefined split do not describe the supplied 5,000-row CSV.

Stanford distributes pretrained GloVe vectors under the Public Domain Dedication and License v1.0. The GloVe software has separate Apache 2.0 terms. External vectors are not included in the repository.
