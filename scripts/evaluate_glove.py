"""Optional GloVe analogy, nearest-neighbor, PCA, and t-SNE experiment."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib.pyplot as plt
import pandas as pd
from threadpoolctl import threadpool_limits

from laml.experiments import ANALOGIES, RESULTS, savefig, setup, write_json
from laml.text import evaluate_analogies, load_glove, project_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--dimension',type=int,default=300)
    parser.add_argument('--limit',type=int,default=None,help='Optional vocabulary prefix; changes the candidate vocabulary and analogy scores')
    args = parser.parse_args()
    setup()
    with threadpool_limits(limits=1):
        space = load_glove(args.file,args.dimension,args.limit)
        metrics = evaluate_analogies(space,ANALOGIES)
        metrics.update({'vocabulary_size':len(space.words),'dimensions':args.dimension,'vocabulary_limit':args.limit,
                        'input_file':args.file.name,'status':'completed'})
        for a,b in [('france','tehran'),('france','paris')]:
            if a in space.index and b in space.index:
                metrics[f'cosine_{a}_{b}'] = float(space.unit[space.index[a]] @ space.unit[space.index[b]])
        if 'france' in space.index:
            metrics['france_neighbors'] = space.neighbors('france')
        write_json('glove_metrics.json',metrics)
        chosen = [w for w in ['tall','taller','tallest','big','bigger','biggest','wide','wider','widest','man','woman','king','queen'] if w in space.index]
        if len(chosen) >= 4:
            vectors = space.vectors[[space.index[w] for w in chosen]]
            for method in ['pca','tsne']:
                projected = project_vectors(vectors,method)
                fig,ax = plt.subplots(figsize=(7,4))
                ax.scatter(projected[:,0],projected[:,1],color='#157f83')
                for word,(x,y) in zip(chosen,projected):
                    ax.annotate(word,(x,y),xytext=(4,3),textcoords='offset points',fontsize=9)
                ax.set(title=f'GloVe word vectors: {method.upper()}',xlabel='Coordinate 1',ylabel='Coordinate 2')
                savefig(f'glove_{method}.png',fig)
        print(f"Analogy coverage {metrics['covered']}/{metrics['total']}; covered accuracy {metrics['accuracy_covered']}")


if __name__ == '__main__':
    main()
