# This is a short snippet reading and normalizing the json files.

import pandas as pd
import json

with open("clasrep_bert.json") as f:
  param = json.load(f)
results = pd.io.json.json_normalize(param)
results.mean()
