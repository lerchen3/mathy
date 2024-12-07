import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

math = pd.read_csv('filtered_problems (1).csv')

math.to_csv('obscure_math.csv', index=False)
