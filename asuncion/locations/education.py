import numpy as np
import pandas as pd

"""
Yield education location candidates.
"""

def configure(context):
    context.stage("asuncion.data.education")


def execute(context):
    return context.stage("asuncion.data.education")
