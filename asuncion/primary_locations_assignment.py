import pandas as pd
import numpy as np

def configure(context):
    education_method = context.config("education_assignment_method")
    if education_method == "gravity":
        context.stage("asuncion.gravity.education", alias = "education_stage")
    elif education_method == "distance":
        context.stage("asuncion.distance.education", alias = "education_stage")
    else:
        raise RuntimeError("Unknown stage: %s" % education_method)

    work_method = context.config("work_assignment_method")
    if work_method == "gravity":
        context.stage("asuncion.gravity.work", alias = "work_stage")
    elif work_method == "distance":
        context.stage("asuncion.distance.work", alias = "work_stage")
    else:
        raise RuntimeError("Unknown stage: %s" % work_method)


def execute(context):
    df_education = context.stage("education_stage")
    df_work = context.stage("work_stage")
    return df_work, df_education