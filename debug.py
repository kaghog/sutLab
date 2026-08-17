# debug.py
import pickle
import sys
import pandas as pd
import numpy as np
import geopandas as gpd

def snapshot(context, configs=(), stages=(), path="../export/snapshot.pkl"):
    data = {
        "config": {x: context.config(x) for x in configs},
        "stage": {x: context.stage(x) for x in stages},
    }
    data["config"]["debug"] = True
    with open(path, "wb") as f:
        pickle.dump(data, f)
    sys.exit(0)

class DebugContext:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.snapshot = pickle.load(f)

    def config(self, name, *args, **kwargs):
        return self.snapshot["config"][name]

    def stage(self, name, *args, **kwargs):
        return self.snapshot["stage"][name]

#
"""

class DebugContext:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.snapshot = pickle.load(f)

    def config(self, name, *args, **kwargs):
        return self.snapshot["config"][name]

    def stage(self, name, *args, **kwargs):
        return self.snapshot["stage"][name]

if __name__ == "__main__":
    context = DebugContext("../export/snapshot.pkl")
    execute(context)

    

# for export
    from debug import snapshot
    snapshot(context,
        configs=["output_path", "data_path", "analysis_path", "output_prefix"],
        stages=["synthesis.output", "seville.gravity.od_zones"]
        )

"""