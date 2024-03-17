import pandas as pd
from numpy import random


class FixedPortfolio:
	def __init__(self, quotes, weights=None, name="fixed portfolio"):
		"""
		Parameters
		----------
		quotes : DataFrame
		weights : dict from str to a number 
			keys represent instrument, values represent their relative weights is the portfolio
		name : str
		"""
		if weights:
			self._weights = pd.Series(weights, name=name)
			self._weights /= self._weights.sum()

			self.universe = sorted(weights.keys())
		else:
			self.universe = list(quotes.columns)
			self._weights = pd.Series(1 / len(self.universe), index=self.universe)

		self.name = name
		self._quotes = quotes[self.universe].sort_index()
		return

	def get_quotes(self):
		return self._quotes

	def get_returns(self, start=None, end=None):
		"""
		Parameters
		----------
		start : str
			the date in YYYY-MM-DD format
		end : str
			the date in YYYY-MM-DD format
		"""
		if start and end:
			return self._quotes.loc[start:end].pct_change().multiply(self._weights, axis=1).sum(axis=1).rename(self.name)
		return self._quotes.pct_change().multiply(self._weights, axis=1).sum(axis=1).rename(self.name)


class DynamicPortfolio(FixedPortfolio):
	def __init__(self, quotes, name="dynamic portfolio"):
		self._quotes = quotes
		self.universe = list(quotes.columns)
		self.name = name

		self._weights = pd.DataFrame([], index=self._quotes.index, columns=self.universe)

	def _get_weights(self, date: str | None=None):
		return pd.Series(1 / len(self.universe), index=self.universe)

	def get_weights(self, date: str | None=None):
		if pd.isna(self._weights.loc[date].iloc[0]):
			self._weights.loc[date] = self._get_weights(date)
		return self._weights.loc[date]

	def get_returns(self, start=None, end=None):
		if start:
			if not end:
				end = start
				start = self._quotes.index[self._quotes.index.get_loc(start)-1]	
			pure_returns = self._quotes.loc[start:end].fillna(0).pct_change()
		else:
			pure_returns = self._quotes.fillna(0).pct_change()
		return (pure_returns * pd.DataFrame(pure_returns.index.map(self.get_weights).tolist(), index=pure_returns.index).shift(1)).sum(axis=1)[1:].rename(self.name)


class RandomlyWeightedPortfolio(DynamicPortfolio):
	def _get_weights(self, date: str | None=None):
		weights = random.rand(len(self.universe))
		weights /= weights.sum()
		return pd.Series(weights, index=self.universe)


if __name__ == "__main__":
	raise RuntimeError("This is a module!")