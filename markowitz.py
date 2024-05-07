import numpy as np
import pandas as pd
from scipy.optimize import fminbound


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
	n = covariance.shape[0]
	M = np.append( np.append(covariance, np.ones((1, n)), axis=0), -np.ones((n+1, 1)), axis=1 )
	M[n, n] = 0

	v = np.zeros(n+1)
	v[n] = 1
	return np.linalg.solve(M, v)[:-1]


def mv(covariance: np.ndarray, expected_returns, desired_return):
	"""
	Parameters
	----------
	cov : matrix
		a covariance matrix
	expected_returns : vector
	desired_return : float

	Returns
	-------
	Vector
	"""
	M = np.zeros(np.add(covariance.shape, 2))
	M[:-2, :-2] = covariance
	M[-2, :-2] = 1
	M[:-2, -2] = -1
	M[-1, :-2] = expected_returns.values
	M[:-2, -1] = -expected_returns.values

	v = np.zeros(covariance.shape[0]+2)
	v[-2] = 1
	v[-1] = desired_return

	return np.linalg.solve(M, v)[:-2]


def ms(covariance: np.ndarray, expected_returns, min_return=0., max_return=.1, tol=1E-4, maxiter:int=2**6):
	def sharp(ret):
		weights = mv(covariance, expected_returns, ret)
		return -ret / (weights @ covariance @ weights)
	ideal_ret = fminbound(sharp, min_return, max_return, xtol=tol, maxfun=maxiter)
	return mv(covariance, expected_returns, ideal_ret), ideal_ret, -sharp(ideal_ret)


def mvs(covariance: np.ndarray, expected_returns, desired_sharpe):
	"""
	Parameters
	----------
	cov : matrix
		a covariance matrix
	expected_returns : vector
	desired_sharpe : float

	Returns
	-------
	Vector
	"""
	M = np.zeros( (covariance.shape[0]+1, covariance.shape[0]+2) )
	M[:-1, :-2] = covariance
	M[-1, :-2] = 1
	M[:-1, -2] = -1
	M[:-1, -1] = -expected_returns.values

	print(pd.DataFrame(covariance))
	print(expected_returns)
	print(pd.DataFrame(M))

	return 

	v = np.zeros(covariance.shape[0]+2)
	v[-2] = 1
	v[-1] = desired_return

	return np.linalg.solve(M, v)[:-2]



class GMVPortfolio(DynamicPortfolio):
	def __init__(self, quotes, window=22, name="markowitz GMV portfolio"):
		super().__init__(quotes, name)
		self._window = window
		self._weights = self._weights.iloc[self._window:]
		self._predicted_returns = self._weights.iloc[self._window:].copy(deep=True)
		return

	def _predict_returns(self, date, start=None):
		if not start:
			start = self._quotes.index.get_loc(date)-self._window
		start = self._quotes.index[start]
		return self._quotes.loc[start:date].dropna(axis=1).pct_change().mean()

	def _get_covariance(self, date: str, start=None):
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

	def predict_returns(self, date, start=None):
		if pd.isna(self._predicted_returns.loc[date].iloc[0]):
			rets = pd.Series(0., self.universe)
			temp = self._predict_returns(date, start)
			rets.loc[temp.index] = temp
			self._predicted_returns.loc[date] = rets
		return self._predicted_returns.loc[date]

	def get_covariance(self, date, start=None):
		return self._get_covariance(date, start)


class MVPortfolio(GMVPortfolio):
	def __init__(self, desired_return, quotes, window=22, name="markowitz MV portfolio"):
		super().__init__(quotes, window, name)
		self._desired_return = desired_return
		return

	def _get_weights(self, date=None):
		weights = pd.Series(0., self.universe)
		if (start:=self._quotes.index.get_loc(date)-self._window) < 0:
			return weights
		cov = self._get_covariance(date, start)
		predicted_returns = self.predict_returns(date, start).loc[cov.index]
		weights.loc[cov.index] = mv(cov, predicted_returns, self._desired_return)
		return weights


class MSPortfolio(GMVPortfolio):
	def __init__(self, quotes, window, name="markowitz MS portfolio"):
		super().__init__(quotes, window, name)
		self._predicted_sharps = pd.Series(0., index=self._weights.index[self._window:])
		return

	def _get_weights(self, date=None):
		weights = pd.Series(0., self.universe)
		if (start:=self._quotes.index.get_loc(date)-self._window) < 0:
			return weights
		cov = self._get_covariance(date, start)
		predicted_returns = self._predict_returns(date, start)
		weights.loc[cov.index], _, self._predicted_sharps[date] = ms(cov, predicted_returns)
		return weights


class MVSPortfolio(GMVPortfolio):
	def __init__(self, desired_sharpe, quotes, window=22, name="markowitze MVS portfolio"):
		super().__init__(quotes, window, name)
		self._desired_sharpe = desired_sharpe
		return

	def _get_weights(self, date=None):
		weights = pd.Series(0., self.universe)
		if (start:=self._quotes.index.get_loc(date)-self._window) < 0:
			return weights
		cov = self._get_covariance(date, start)
		predicted_returns = self.predict_returns(date, start).loc[cov.index]
		weights.loc[cov.index] = mvs(cov, predicted_returns, self._desired_sharpe)
		return weights