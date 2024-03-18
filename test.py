import pandas as pd
import pickle

DATA_DIR = '/home/rds/Desktop/git_ws//sign-korean-language/data/'

data_dict = pickle.load(open('/home/rds/Desktop/git_ws//sign-korean-language/data/tmp_0/30.pickle', 'rb'))

df = pd.DataFrame(data_dict)

print(df.head(40))
