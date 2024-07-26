import json
import numpy as np

with open('Training/vid2player3d/tennis_data/manifest.json', 'r') as f:
    manifest = json.load(f)

print(len(manifest[0]['sequences']['fg']))

names = [seq['clip'] for seq in manifest[0]['sequences']['fg']]

unique_names = set(names)

print(len(unique_names))