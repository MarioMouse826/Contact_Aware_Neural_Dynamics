from lerobot.datasets.lerobot_dataset import LeRobotDataset

# We only look at the 'features' to avoid a massive download
dataset = LeRobotDataset("lerobot/droid_1.0.1")
print(dataset.features)