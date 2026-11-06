import unittest
import importlib.util
from marketing_probe.learned_embedding import project

@unittest.skipUnless(importlib.util.find_spec('numpy'), 'numpy learning extra unavailable')
class EmbeddingBaselineTests(unittest.TestCase):
    def test_projection_is_signed_and_reproducible(self):
        import numpy as np
        data={"train":np.array([[[1.,0.,0.],[0.,1.,0.]]]),
              "val":np.array([[[0.,1.,0.],[1.,0.,0.]]]),
              "test":np.array([[[0.,0.,1.],[0.,1.,0.]]])}
        a=project(data,2,seed=7)
        b=project(data,2,seed=7)
        np.testing.assert_allclose(a["train"],b["train"])
        np.testing.assert_allclose(a["train"],-a["val"])
