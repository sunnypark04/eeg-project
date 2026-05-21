from pathlib import Path
import yaml
import os

# Print Current Directory
print(f"Current directory is {Path.cwd()}")

# Check imports
from my_dl_project.models.simple_cnn import SimpleCNN
from my_dl_project.training.trainer import evaluate, train_one_epoch
from my_dl_project.data.defungi import DEFUNGI_CLASSES, DeFungiDataset

print("SUCCESS: package imports worked")
print(f"SUCCESS: found classes: {DEFUNGI_CLASSES}")

# Check if you can get a data object
data_dir = "/gpfs/projects/dsci410_510/Lab6/defungi/"
dataset = DeFungiDataset(data_dir, transform=None)

print(f"SUCCESS: created DeFungiDataset from {data_dir}")
print(f"SUCCESS: dataset has {len(dataset)} images")

image, label = dataset[0]

print("SUCCESS: loaded the first dataset item")
print(f"SUCCESS: image type = {type(image)}")
print(f"SUCCESS: image size = {image.size}")
print(f"SUCCESS: label tensor = {label}")
print(f"SUCCESS: label value = {label.item()}")

# Check default config can be parsed
config_path = Path("configs/default.yaml")

with config_path.open("r") as f:
    config = yaml.safe_load(f)

print(f"SUCCESS: parsed config file at {config_path}")
print(f"SUCCESS: config keys = {list(config.keys())}")

# Check model can be created
model = SimpleCNN(num_classes=len(DEFUNGI_CLASSES))

print("SUCCESS: created SimpleCNN model")
print(f"SUCCESS: model has {len(DEFUNGI_CLASSES)} output classes")

username = os.environ.get("USER", "unknown_user")
results_path = Path(f"/gpfs/projects/dsci410_510/Lab6/results/{username}.txt")

if results_path.exists():
    print(f"SUCCESS: results file exists at {results_path}")
else:
    print(f"WARNING: results file not found at {results_path}")

 
