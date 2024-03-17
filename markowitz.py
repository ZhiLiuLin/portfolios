import numpy as np
import pandas as pd
from qpsolvers import solve_qp
from scipy import sparse

from .base import DynamicPortfolio


def gmv(covariance: np.ndarray):
	"""
	Parameters
	----------
	cov : matrix
		a covariance matrix

	Returns
	-------
	Vector
	"""
	# n = covariance.shape[0]
	# M = np.append( np.append(covariance, np.ones((1, n)), axis=0), -np.ones((n+1, 1)), axis=1 )
	# M[n, n] = 0

	# v = np.zeros(n+1)
	# v[n] = 1
	# current = np.linalg.solve(M, v)[:-1]

	# while any(current < 0):
	# 	for i, w in enumerate(current):
	# 		if w < 0:
	# 			M[i, :] = 0
	# 			M[:, i] = 0
	# 			M[i, i] = 1
	# 	current = np.linalg.solve(M, v)[:-1]
	# return current

	return solve_qp(sparse.csc_matrix(covariance),
					q=np.zeros( (n:=covariance.shape[0]) ), 
					A=sparse.csc_matrix(np.ones(n)), 
					b=np.ones(1), 
					lb=np.zeros(n), 
					ub=np.ones(n), 
					solver="clarabel")


def mv(covariance: np.ndarray, expected_returns, desired_return):
	"""
	Parameters
	----------
	cov : matrix
		a covariance matrix

	Returns
	-------
	Vector
	"""
	return solve_qp(sparse.csc_matrix(covariance),
					q=np.zeros( (n:=covariance.shape[0]) ), 
					A=sparse.csc_matrix( np.array([np.ones(n), expected_returns]) ), 
					b=np.array([1, desired_return]), 
					lb=np.zeros(n), 
					ub=np.ones(n), 
					solver="clarabel")


# def max_sharp(covariance: np.ndarray, expected_returns, risk_free=0):
# 	def f(x):
# 		weights = mv(covariance, expected_returns, x)
# 		return -(expected_returns @ weights - risk_free) / (weights @ covariance @ weights)
# 	expected_return = -fmin(f, np.mean(expected_returns))[0]
# 	return mv(covariance, expected_returns, expected_return)


class GMVPortfolio(DynamicPortfolio):
	def __init__(self, quotes, window=22, name="markowitz GMV portfolio"):
		super().__init__(quotes, name)
		self._window = window
		return

	def _get_covariance(self, date: str | None, start=None):
		if not start:
			start = self._quotes.index.get_loc(date)-self._window
		start = self._quotes.index[start]
		return self._quotes.loc[start:date].dropna(axis=1).pct_change().iloc[1:].cov()

	def _get_weights(self, date: str | None=None):
		weights = pd.Series(0., self.universe)
		if (start:=self._quotes.index.get_loc(date)-self._window) < 0:
			return weights
		cov = self._get_covariance(date, start)
		weights.loc[cov.index] = gmv(cov.values) 
		return weights

	def get_covariance(self, date, start=None):
		return self._get_covariance(date, start)