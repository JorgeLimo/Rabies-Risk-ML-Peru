import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'ndvi':            np.random.uniform(0.1, 0.9, n),
    'temperature':     np.random.uniform(18, 32, n),
    'precipitation':   np.random.uniform(50, 400, n),
    'forest_loss':     np.random.uniform(0, 1, n),
    'bat_occurrence':  np.random.randint(0, 10, n),
    'population_density': np.random.uniform(1, 500, n),
    'dist_to_forest':  np.random.uniform(0, 50, n),
    'target':          np.random.randint(0, 2, n)  # 1=riesgo alto, 0=riesgo bajo
})

df.to_csv('data/rabies_data.csv', index=False)
print(f"Dataset created: {len(df)} records, {df['target'].sum()} positive cases")