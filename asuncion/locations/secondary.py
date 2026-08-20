import numpy as np
import pandas as pd

"""
Yield education location candidates.
"""

def configure(context):
    context.stage("asuncion.data.work")


def execute(context):
    # TODO: filter out industry and agriculture locations
    raise NotImplementedError
    return context.stage("asuncion.data.work")
