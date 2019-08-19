import unittest
import dynet as dy
import numpy as np

class TestVanillaLSTM(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.rnn = dy.VanillaLSTMBuilder(2, 10, 10, self.m)

    def test_get_parameters(self):
        dy.renew_cg()
        self.rnn.initial_state()
        P_p = self.rnn.get_parameters()
        P_e = self.rnn.get_parameter_expressions()
        for l_p, l_e in zip(P_p, P_e):
            for w_p, w_e in zip(l_p, l_e):
                self.assertTrue(np.allclose(w_e.npvalue(), w_p.as_array()))

    def test_get_parameters_sanity(self):
        self.assertRaises(
            ValueError, lambda x: x.get_parameter_expressions(), self.rnn)

    def test_initial_state_vec(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        self.rnn.initial_state(init_s)
        self.assertTrue(True)

    def test_set_h(self):
        dy.renew_cg()
        init_h = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_h(init_h)
        self.assertTrue(True)

    def test_set_c(self):
        dy.renew_cg()
        init_c = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_c)
        self.assertTrue(True)

    def test_set_s(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_s)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()