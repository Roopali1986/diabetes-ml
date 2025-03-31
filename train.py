from azureml.core import Workspace, Dataset
from sklearn.linear_model import Ridge
import joblib

# Connect to workspace
ws = Workspace.from_config()

# Load Diabetes_1 dataset
dataset = Dataset.get_by_name(ws, name='Diabetes_1')
df = dataset.to_pandas_dataframe()

# Train simple model
X = df.drop("target", axis=1)
y = df["target"]

model = Ridge(alpha=0.5)
model.fit(X, y)

# Save model
joblib.dump(model, 'outputs/model.pkl')
