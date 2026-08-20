import numpy as np
import pandas as pd

"""
Yield education location candidates for Germany.
"""

def configure(context):
    context.stage("seville.data.education.merged")


def execute(context):
    return context.stage("seville.data.education.merged")
