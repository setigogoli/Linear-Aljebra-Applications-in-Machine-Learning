"""Independent numerical checks for gradients and data handling."""
import unittest
from pathlib import Path
import tempfile

import numpy as np

from laml.models import CharacterRNN, LSTMClassifier, MLP, MatrixFactorization, least_squares, softmax
from laml.text import VectorSpace, cooccurrence, ppmi, load_glove, project_vectors


def max_gradient_error(model, loss_call):
    _, analytic, *_ = loss_call()
    errors = []
    for key, array in model.params.items():
        for index in np.ndindex(array.shape):
            old = array[index]
            array[index] = old + 1e-5
            plus = loss_call()[0]
            array[index] = old - 1e-5
            minus = loss_call()[0]
            array[index] = old
            errors.append(abs((plus-minus)/2e-5 - analytic[key][index]))
    return max(errors)


class NumericalTests(unittest.TestCase):
    def test_softmax_large_logits(self):
        p = softmax(np.array([[10000., 10001.], [-10000., -10001.]]))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1)

    def test_least_squares(self):
        self.assertAlmostEqual(least_squares(np.arange(6), 3*np.arange(6)), 3)

    def test_mlp_gradient(self):
        x = np.array([[.7, -.4], [-.3, .8], [1.2, .2]])
        model = MLP(hidden_dim=3)
        self.assertLess(max_gradient_error(model, lambda: model.loss_grad(x, np.array([0, 1, 1]))), 1e-7)

    def test_rnn_gradient(self):
        model = CharacterRNN(vocab_size=4, hidden_size=3)
        self.assertLess(max_gradient_error(model, lambda: model.loss_grad([0, 1, 2], [1, 2, 3])), 1e-7)

    def test_lstm_gradient_with_padding(self):
        model = LSTMClassifier(input_dim=2, hidden_dim=2)
        x = np.random.default_rng(5).normal(size=(2, 3, 2))
        self.assertLess(max_gradient_error(model, lambda: model.loss_grad(x, np.array([3, 1]), np.array([1, 0]))), 1e-7)

    def test_lstm_padding_invariance(self):
        model = LSTMClassifier(2, 3)
        x = np.random.default_rng(5).normal(size=(2, 3, 2))
        padded = np.concatenate([x, np.ones((2, 2, 2))*1000], axis=1)
        np.testing.assert_allclose(model.forward(x, np.array([3, 1]))[0], model.forward(padded, np.array([3, 1]))[0])

    def test_sparse_context_window(self):
        words, c, _ = cooccurrence([['a', 'b', 'c']], min_count=1, window=1)
        np.testing.assert_array_equal(c.toarray(), [[0,1,0], [1,0,1], [0,1,0]])
        self.assertTrue((ppmi(c).data >= 0).all())

    def test_analogy_excludes_inputs(self):
        space = VectorSpace(['a','b','c','d'], [[1,0],[1,1],[0,1],[0,2]])
        self.assertEqual(space.analogy('a','b','c')[0][0], 'd')

    def test_factorization_update_uses_old_factors(self):
        model = MatrixFactorization(1,1,rank=2)
        p, q = model.p[0].copy(), model.q[0].copy()
        error = 4 - p@q
        model.epoch(np.array([0]), np.array([0]), np.array([4.]), np.random.default_rng(1))
        np.testing.assert_allclose(model.p[0], p + .005*(error*q-.02*p))
        np.testing.assert_allclose(model.q[0], q + .005*(error*p-.02*q))

    def test_glove_loader_and_projections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vectors.txt'
            path.write_text('man 1 0\nwoman 0 1\nking 2 0\nqueen 1 1\n')
            space = load_glove(path, dimension=2)
            self.assertEqual(space.analogy('man','woman','king')[0][0], 'queen')
            self.assertEqual(project_vectors(space.vectors,'pca').shape,(4,2))


if __name__ == '__main__':
    unittest.main()
