import pandas as pd


def adjust_quotes(quotes, adj_factor):
	return ( (temp:=adj_factor.cumprod()) / temp.iloc[-1]) * quotes