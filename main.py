import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
store_data = pd.read_csv('World-Cups-After-Modification .csv', header=None)
teams = store_data.iloc[:,2:6].values.tolist()
teams.pop(0)
te = TransactionEncoder()
data = te.fit(teams).transform(teams)
df = pd.DataFrame(data, columns=te.columns_)
df = df.replace(False,0)
df = df.replace(True,1)
df = apriori(df, min_support = 0.1,use_colnames = 1, verbose = 1)
print(df)
df_ar = association_rules(df, metric = "confidence", min_threshold = 0.1)
print(df_ar)
#################################################################################
print('#################################################################################')
continent = store_data.iloc[:,10:15].values.tolist()
continent.pop(0)
te = TransactionEncoder()
data = te.fit(continent).transform(continent)
df = pd.DataFrame(data, columns=te.columns_)
df = df.replace(False,0)
df = df.replace(True,1)
df = apriori(df, min_support = 0.1,use_colnames = 1, verbose = 1)
print(df)
df_ar = association_rules(df, metric = "confidence", min_threshold = 0.1)
print(df_ar)
