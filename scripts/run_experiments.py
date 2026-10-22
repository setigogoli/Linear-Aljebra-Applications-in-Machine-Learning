"""Run from any working directory; the repository supplies all paths."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from laml.experiments import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['all','neural','embeddings','sentiment','recommender'],default='all')
    run(parser.parse_args().part)
