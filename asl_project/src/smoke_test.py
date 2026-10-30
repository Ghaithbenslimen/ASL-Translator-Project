import sys
sys.path.insert(0, "/content/src")
from train import CONFIG, train
CONFIG["epochs"] = 2
train()
