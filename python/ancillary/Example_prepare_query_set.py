import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from set_addition import prepare_query_set
import pandas as pd

df = pd.read_csv(r"ancillary\set_addition\hierarchy_2.csv")

members = {"Geography": ['LA', 'NYC', 'Germany','Canada', 'Vancouver'], 'Product': ['iPhone', 'Toothpaste', 'Toiletries', 'Laptops']}

result= prepare_query_set(df, members)

result.to_clipboard(index=False)

print('Results copied to clipboard')
