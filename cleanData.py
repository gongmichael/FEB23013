import pandas as pd
import numpy as np

# load data and parse the integer date to datetime object
data = pd.read_csv('data/sample.zip', parse_dates=[1])

# rename the variables to make data look cleaner
data.rename(columns={'impl_volatility': 'imvol'}, inplace=True)
data.rename(columns={'days': 'maturity'}, inplace=True)

# drop unnecessary varibles
data.drop(columns=['index_flag','secid','sic'], axis=1, inplace=True)

# maturity filter
data = data[data['maturity'] <= 365].copy()
# transform the delta of put to call equivalent delta
data['delta'] = data['delta'] + 100*data['cp_flag'].map({'C':0,'P':1})

# sort data
toSort = ['ticker', 'date', 'cp_flag', 'maturity', 'delta']
data.sort_values(by=toSort, inplace=True)

# casting the data to float and smaller integer (save space and RAM!)
select = ['maturity', 'delta']
data[select] = data[select].astype(np.int16)
data['imvol'] = data['imvol'].astype(np.float32)

# Save data to feather format by ticker (for fast read and easier access)
# feather format is the state of art storage format, very fast to read!
import os 
if not os.path.exists('data/database/'): #create directory is not exists
    os.mkdir('data/database/')
tickers = data.ticker.unique()
for ticker in tickers:
    # reset the index so the index for each ticker start from 0
    tmp = data[data['ticker'] == ticker].reset_index(drop=True)
    tmp.to_csv(f'data/database/{ticker}.csv',index=False) # the f'{}' is the functional string, powerful tool!

# later you can read the data by pd.read_feather(f'data/database/{ticker}.ft')