import numpy as np

"""
This stage exports synthetic population from IPU.
"""


def configure(context):
    context.stage("seville.ipu.attributed")
    context.config("analysis_path")


def execute(context):
    df = context.stage("seville.ipu.attributed")           
    df.to_csv(f"{context.config('analysis_path')}/synthetic_population.csv", sep = ";", index = None, lineterminator = "\n")
