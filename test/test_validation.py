import unittest
import numpy as np
from casadi import SX, horzcat, vertcat
import nosnoc
from examples.sliding_mode_ocp import get_default_options


class TestValidation(unittest.TestCase):
    """Test validation of the inputs."""

    def test_no_c_S_and_g_Stewart(self):
        x = SX.sym('x')
        F = [horzcat(-1, 1)]
        c = vertcat(0)
        S = [np.array([[1], [-1]])]
        g = [x, -x]
        X0 = np.array([0])
        with self.assertRaises(ValueError):
            nosnoc.NosnocModel(x=x, F=F, x0=X0, c=c, S=S, g_Stewart=g)

    def test_no_switching_system(self):
        x = SX.sym('x')
        F = [horzcat(-1)]
        c = vertcat(1)
        S = [np.array([-1])]
        X0 = np.array([0])
        model = nosnoc.NosnocModel(x=x, F=F, x0=X0, c=c, S=S)
        opts = get_default_options()
        with self.assertRaises(Warning):
            nosnoc.NosnocSolver(opts, model)

    def test_switching_system_wrong_g_Stewart(self):
        x = SX.sym('x')
        F = [horzcat(-1, 1)]
        g = [x, -x, -100]
        X0 = np.array([0])

        model = nosnoc.NosnocModel(x=x, F=F, x0=X0, g_Stewart=g)
        opts = get_default_options()
        with self.assertRaises(ValueError):
            nosnoc.NosnocSolver(opts, model)

    def test_switching_system_wrong_S_shape(self):
        x = SX.sym('x')
        F = [horzcat(-x, 2 * x)]
        c = [x + x**2]
        S = [np.array([-1])]
        X0 = np.array([0])

        model = nosnoc.NosnocModel(x=x, F=F, x0=X0, c=c, S=S)
        opts = get_default_options()
        with self.assertRaises(ValueError):
            nosnoc.NosnocSolver(opts, model)


if __name__ == "__main__":
    unittest.main()
