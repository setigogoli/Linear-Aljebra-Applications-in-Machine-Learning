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
