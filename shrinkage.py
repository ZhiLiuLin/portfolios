from .markowitz import GMVPortfolio
from .factors import FactorPortfolio


class FactorGMVPortfolio(FactorPortfolio, GMVPortfolio):
	def __init__(self, quotes, factors, risk_free, window, name="factor GMV portfolio"):
		super().__init__(quotes=quotes, factors=factors, risk_free=risk_free, window=window)
		self.name = name
		return


class ShrinkageGMVPortfolio(FactorPortfolio, GMVPortfolio):
	def __init__(self, quotes, factors, risk_free, lamb, window, name="factor shrinkage GMV portfolio"):
		super().__init__(quotes=quotes, factors=factors, risk_free=risk_free, window=window)
		self.name = name
		self.lamb = lamb
		return

	def _get_covariance(self, date, start=None):
		return self.lamb * FactorPortfolio._get_covariance(self, date, start) + (1 - self.lamb) * GMVPortfolio._get_covariance(self, date, start)