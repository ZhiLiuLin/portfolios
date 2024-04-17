import numpy as np
import pandas as pd

from .base import DynamicPortfolio


class FactorPortfolio(DynamicPortfolio):
	def __init__(self, quotes, factors, risk_free, window, name="factor portfolio"):
		super().__init__(quotes, name=name)
		self._window = window
		self._factors = factors.loc[self._quotes.index]
		self._factors.insert(0, "alpha", 1)

		self._excess_returns = self._quotes.pct_change(fill_method=None).iloc[1:]
		self._excess_returns = self._excess_returns.sub(risk_free[self._excess_returns.index], axis=0)

		self._betas = pd.DataFrame(np.nan, index=pd.MultiIndex.from_product([self._quotes.index, self._factors.columns]), columns=self.universe)
		return

	def _get_betas(self, date, start=None):
		if not start:
			if (start:=self._quotes.index.get_loc(date)-self._window+1) < 1:
				return 0
			start = self._quotes.index[start]

		betas = self._betas.loc[date].fillna(0)

		Y = self._excess_returns.loc[start:date].dropna(axis=1)
		X = self._factors.loc[start:date]
		betas[Y.columns] = (np.linalg.inv(X.T @ X) @ X.T @ Y).set_index(self._factors.columns)
		return betas

	def get_betas(self, date, start=None):
		if pd.isna(self._betas.loc[date].iloc[0].iloc[0]):
			self._betas.loc[date] = self._get_betas(date, start).values
		return self._betas.loc[date]

	def _get_covariance(self, date, start=None):
		if (start:=self._quotes.index.get_loc(date)-self._window+1) < 1:
			return 0
		start = self._quotes.index[start]

		excess_rets = self._excess_returns.loc[start:date].dropna(axis=1)

		betas = self.get_betas(date, start)[excess_rets.columns]
		X = self._factors.loc[start:date]

		M = betas.drop("alpha").mul( X.drop(columns="alpha").std() , axis="rows")

		return M.T @ M + np.diagflat( (excess_rets - X @ betas).var() )