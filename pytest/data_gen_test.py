import unittest

import numpy as np

import py.algos as alg
import py.data_gen as dg


class DataGenTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cmd = f'--num-applicants 1000 --applicants-per-round 100 --clip --envs-accept-rates .25 --rank-type uniform --num-envs 2'
    args = alg.get_args(cmd)
    np.random.seed(1)
    data_v1 = dg.generate_data(
      args.num_applicants, args.applicants_per_round, args.fixed_effort_conversion, args
    )

    data_v2 = dg.generate_data_v2(
      args.num_applicants, args.applicants_per_round, args.fixed_effort_conversion, args
    )

    # assignment
    cls.data_v1 = data_v1
    cls.data_v2 = data_v2

    return

  def test_marginal_data(self):
    data_v1 = self.data_v1
    data_v2 = self.data_v2

    # test base covariates
    idx = 0
    rtol = 0.02
    np.testing.assert_allclose(data_v2[idx].mean(axis=0), data_v1[idx].mean(axis=0), rtol=rtol)
    np.testing.assert_allclose(data_v2[idx].mean(axis=0), [900, 2.025], rtol=rtol)
    self.assertRaises(
      AssertionError,
      np.testing.assert_allclose,
      actual=data_v2[idx].mean(axis=0), desired=[1000, 2.25], rtol=rtol
    )

    # test environmental noise o, (n,T)
    idx = 10
    rtol = 0.03
    np.testing.assert_allclose(data_v2[idx].mean(axis=1), data_v1[idx].mean(axis=1), rtol=rtol)
    np.testing.assert_allclose(data_v2[idx].mean(axis=1), [1, 1], rtol=rtol)
    self.assertRaises(
      AssertionError,
      np.testing.assert_allclose,
      actual=data_v2[idx].mean(axis=1), desired=[2, 2], rtol=rtol
    )

    # test effort matrix e, ()
    idx = 3
    rtol = 0.02
    np.testing.assert_allclose(data_v2[idx], data_v1[idx], rtol=rtol)
    np.testing.assert_allclose(data_v2[idx], [[100, 0], [0, 1]], rtol=rtol)
    self.assertRaises(
      AssertionError,
      np.testing.assert_allclose,
      actual=data_v2[idx], desired=[[110, 0], [0, 1.5]], rtol=rtol
    )

    return

  def test_strategic_variables(self):
    # because thetas have large variance,
    # having scaled duplicates skews things
    cmd = f'--num-applicants 1000 --applicants-per-round 1 --clip --envs-accept-rates .25 --rank-type uniform --num-envs 2'
    args = alg.get_args(cmd)
    np.random.seed(1)
    data_v1 = dg.generate_data(
      args.num_applicants, args.applicants_per_round, args.fixed_effort_conversion, args
    )

    data_v2 = dg.generate_data_v2(
      args.num_applicants, args.applicants_per_round, args.fixed_effort_conversion, args
    )

    # test thetas, (n,T,m)
    # (thetas have large variance)
    idx = 4
    np.testing.assert_allclose(data_v2[idx].mean(axis=1), data_v1[idx].mean(axis=1), rtol=0.3)
    np.testing.assert_allclose(data_v2[idx].mean(axis=1), [[1, 1], [1, 2]], rtol=0.2)
    np.testing.assert_allclose(data_v1[idx].mean(axis=1), [[1, 1], [1, 2]], rtol=0.2)
    self.assertRaises(
      AssertionError,
      np.testing.assert_allclose,
      actual=data_v2[idx].mean(axis=1), desired=[[2, 2], [2, 1]], rtol=0.3
    )

    # test covariates X, (T,m)
    idx = 1
    rtol = 0.04
    np.testing.assert_allclose(data_v2[idx].mean(axis=0), data_v1[idx].mean(axis=0), rtol=rtol)
    np.testing.assert_allclose(data_v2[idx].mean(axis=0), [1000, 3.4], rtol=rtol)
    self.assertRaises(
      AssertionError,
      np.testing.assert_allclose,
      actual=data_v2[idx].mean(axis=0), desired=[1100, 4], rtol=rtol
    )

    #
    # non-clipping case, with heterogeneous theta-stars,
    # and must have more envs (to stabilise the randomness in theta-stars)
    #
    cmd = f'--num-applicants 1000 --applicants-per-round 1 --envs-accept-rates .25 --theta-star-std 0.1 --rank-type uniform --num-envs 50'
    args = alg.get_args(cmd)
    np.random.seed(1)
    data_v1 = dg.generate_data(
      args.num_applicants, args.applicants_per_round, args.fixed_effort_conversion, args
    )
    data_v2 = dg.generate_data_v2(
      args.num_applicants, args.applicants_per_round, args.fixed_effort_conversion, args
    )

    # test outcomes y, (n,T)
    idx = 2
    rtol = 0.02
    np.testing.assert_allclose(data_v2[idx].mean(), data_v1[idx].mean(), rtol=rtol)
    np.testing.assert_allclose(data_v2[idx].mean(), 15, rtol=rtol)
    np.testing.assert_allclose(data_v2[idx].var(), data_v1[idx].var(), rtol=0.04)
    np.testing.assert_allclose(data_v2[idx].var(), 26, rtol=0.04)

    return


if __name__ == "__main__":
  unittest.main()
