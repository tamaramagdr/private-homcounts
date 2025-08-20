import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

runs = []

for run, tw in zip(range(1, 10), [3, 3, 3, 1, 1, 1, 2, 2, 2]):
        name = 'results/bound_vs_empirical/OGBG-MOLHIV_full_kernel_16_50_run_' + str(run) + '.csv'
        print(name)
        df = pd.read_csv(name)
        df['run'] = run
        df['tw'] = tw
        # add a column called 'graph_id' that is the index of the dataframe
        df['graph_id'] = df.index
        runs.append(df)

# For each of the runs, compute the maximum discrepancy between the empirical values.
con = pd.concat(runs)
range_df = con.groupby(['graph_id']).agg(['max', 'min'])
range_df['discrepancy'] = np.abs(range_df['empirical']['max'] - range_df['empirical']['min'])
# Plot a histogram of the discrepancies, with a bin for zero values specifically.
plt.hist(range_df['discrepancy'], bins=[0, 1e-10] + list(np.linspace(1e-10, range_df['discrepancy'].max(), 100)))
# count how many times discrepancy is zero
zero_count = (range_df['discrepancy'] == 0).sum()
print(zero_count)
plt.show()

print('hi')
