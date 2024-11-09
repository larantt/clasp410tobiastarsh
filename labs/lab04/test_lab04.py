#!/usr/bin/envs python3
"""
Unit test module for lab 4. This can be called to ensure that
the heat difference equation solver is working as expected.
"""
import unittest
import numpy as np
import datetime
from lab04 import heatdiff

class TestHeatdiff(unittest.TestCase):
    """
    Class contains unit tests for the verification step in the 
    lab methodology.

    Adapted from https://www.geeksforgeeks.org/unit-testing-python-unittest/
    """
    def test_heateqn(self):
        """
        Unit test solves for wire test case given in lab code. This has 
        initial conditions of 4*x - 4*(x**2) and boundary conditions set
        to 0C.

        Also creates and saves a heatmap of the solver solution and the
        reference solution to be shown in the lab.

        Parameters
        ----------
        self : Self@TestHeatdiff
            unit test class object
        """
        # Solution to problem 10.3 from fink/matthews as a nested list:
        sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
        [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
        [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
        [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
        [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
        [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
        [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
        [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
        [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
        [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
        [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
        # Convert to an array and transpose it to get correct ordering:
        sol10p3 = np.array(sol10p3).transpose()

        # solve the heat equation for the test case
        x,t,solver_result,_ = heatdiff(1,0.2,0.2,0.02)

        # confirm the coefficients and the temperatures are equal to 1 decimal place
        np.testing.assert_allclose(solver_result, sol10p3, rtol=1E-5)
        # print the date and time of the unit test for reference
        now = datetime.datetime.now()
        print(f'Unit test executed at {now.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    unittest.main()