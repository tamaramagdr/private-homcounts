import pandas as pd

# If needed, clean data by making near duplicates exactly the same for nicer plotting.

df = pd.read_csv("MOLBACE_16_50_all_tw1.csv",
                 header=None)

col = df.shape[1] - 2
df[col] = df[col].round(10)
unique_vals = {}

for i, val in enumerate(df[col]):
    found = False
    for u in unique_vals:
        if abs(val - u) / max(abs(val), abs(u), 1e-10) < 1e-1:
            df.at[i, col] = unique_vals[u]
            found = True
            break
    if not found:
        unique_vals[val] = val

df.to_csv("MOLBACE_16_50_all_tw1_postprocessed.csv",
          index=False, header=False)