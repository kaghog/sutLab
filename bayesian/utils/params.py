# ============================================
# Configuration section (edit the assignments only)
# ============================================
class Config:

    # structure learning
    STRUCTURE_ALGORITHM = "HillClimbSearch"  # 'HillClimbSearch' | 'PC' | 'GES' | 'ExpertInLoop'
    SCORING_METHOD = "aic-d"                 # 'bic-d' | 'aic-d'
    # optional kwargs forwarded into structure learner (e.g., max_indegree for HillClimb)
    STRUCTURE_KWARGS = {} #{"max_indegree": 4}

    # parameter estimation: 'MLE' or 'Bayesian'
    PARAMETER_ESTIMATOR = "MLE"
    PARAM_ESTIMATOR_KWARGS = dict()          # e.g., {'prior_type':'BDeu','equivalent_sample_size':5}

    # core sampling: 'Forward' or 'Gibbs' (this pipeline uses forward sampling by default)
    # core here means Head +/- Spouse + Household data
    CORE_SAMPLING_METHOD = "Forward"

    # dependent members (children / others) sampling default: 'LikelihoodWeighted' or 'Rejection'
    DEPENDENT_SAMPLING_METHOD = "LikelihoodWeighted"
    # multiplier for drawing an over-sample set when using likelihood-weighted sampling
    LWS_MULTIPLIER = 100

    # size: by default, match the real number of core households (1.0). Increase to grow population.
    SAMPLE_SIZE_MULTIPLIER = 1

    # normalize: boolean value if to display relative frequencies (percentages)
    # instead of raw counts in the comparison plots.
    NORMALIZE = True 

    # output options
    SAVE_PER_TYPE = False # Saves for each household type
    SAVE_REGION_WIDE = True # Saves a merged file
    OUTPUT_PREFIX = "SYNTHETIC"
    
    # GR Configuration
    ENABLE_GR_POSTPROCESSING = True  # Set to False to skip GR post-processing
    GR_MAX_ITERATIONS = 500
    GR_TOLERANCE = 1e-8
    GR_TARGET_COLUMNS = None  # None for auto-detection, or specify list like ['Age', 'Sex', 'Residence_type']