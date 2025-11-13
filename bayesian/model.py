import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import networkx as nx
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import *
from pgmpy.sampling import *
from pgmpy.base import DAG
from pgmpy.utils import *

warnings.filterwarnings("ignore")

from collections import defaultdict
from scipy.stats import chi2_contingency

from utils.logging_config import setup_logger
import utils.constants as cons

model_logger = setup_logger("model_logger", "model.log")


def learn_structure(data: pd.DataFrame,
                       scoring_method: str = 'bic-d',
                       algorithm: str = 'HillClimbSearch',
                       start_dag: Optional[DAG] = None,
                       expert_knowledge: Optional[ExpertKnowledge] = None,
                       **kwargs) -> BayesianNetwork:

    if algorithm == 'HillClimbSearch':
        learner = HillClimbSearch(data)
        # Pass start_dag and expert_knowledge if provided
        estimate_kwargs = {'scoring_method': scoring_method}
        if start_dag is not None:
            estimate_kwargs['start_dag'] = start_dag
        if expert_knowledge is not None:
            estimate_kwargs['expert_knowledge'] = expert_knowledge
        estimate_kwargs.update(kwargs)
        model = learner.estimate(**estimate_kwargs)
    elif algorithm == 'PC':
        learner = PC(data)
        estimate_kwargs = {'variant': 'parallel', 'ci_test': 'chi_square', 'return_type': 'dag'}
        if expert_knowledge is not None:
            estimate_kwargs['expert_knowledge'] = expert_knowledge
        estimate_kwargs.update(kwargs)
        model = learner.estimate(**estimate_kwargs)
    elif algorithm == 'GES':
        learner = GES(data)
        estimate_kwargs = {'scoring_method': scoring_method}
        if expert_knowledge is not None:
            estimate_kwargs['expert_knowledge'] = expert_knowledge
        estimate_kwargs.update(kwargs)
        model = learner.estimate(**estimate_kwargs)
    elif algorithm == 'MMHC':
        learner = MmhcEstimator(data)
        estimate_kwargs = {'scoring_method': scoring_method}
        if expert_knowledge is not None:
            estimate_kwargs['expert_knowledge'] = expert_knowledge
        estimate_kwargs.update(kwargs)
        model = learner.estimate(**estimate_kwargs)
    elif algorithm == "ExpertInLoop":
        descriptions = {}
        learner = ExpertInLoop(data)
        estimate_kwargs = {'pval_threshold': 0.05, 'effect_size_threshold': 0.01,
                           'orientation_fn': llm_pairwise_orient,
                           'variable_descriptions': descriptions, 'llm_model': "gpt-3.5-turbo"}
        if expert_knowledge is not None:
            estimate_kwargs['expert_knowledge'] = expert_knowledge
        estimate_kwargs.update(kwargs)
        model = learner.estimate(**estimate_kwargs)
    else:
        raise ValueError(f"Unknown structure algorithm: {algorithm}")

    # Convert DAG to BayesianNetwork if necessary
    if isinstance(model, DAG):
        model = BayesianNetwork(model)

    return model

def print_network_edges(model, name):
    # print edges of BN
    model_logger.info(f"BN Edges for: ({name})")
    print(f"\n--- Edges for: {name} ---")
    for u, v in model.edges():
        model_logger.info(f"  {u} --> {v}")
        print(f"  {u} --> {v}")

def visualize_bn(model, title="Bayesian Network", figsize=(8, 6), save_path='./bn.png'):
    # BN visualization helper
    # if isinstance(model, BayesianNetwork):
    #     G = model.to_directed()
    # else:
    #     G = model

    # n_nodes = len(model.nodes())
    # spacing_factor = max(1.5, np.log10(n_nodes) * 2)
    # k_val = spacing_factor / np.sqrt(n_nodes)

    # base_size = 8 

    # figsize_factor = max(1, np.log(n_nodes) / np.log(10) * 1.5)
    # figsize = (base_size * figsize_factor, base_size * figsize_factor)

    # # pos = nx.spring_layout(G, k=k_val, iterations=100, seed=42)
    # pos = nx.shell_layout(G)
    # fig, ax = plt.subplots(figsize=figsize)
    # node_size = max(100, 1000 / np.log2(n_nodes + 2))
    # font_size = max(8, 20 / np.log2(n_nodes + 2))
    # nx.draw_networkx_nodes(
    #     model, pos,
    #     node_size=node_size,
    #     node_color='skyblue',
    #     edgecolors='black',
    #     alpha=0.8,
    #     ax=ax
    # ) 
    # label_pos = {n: (x, y + 0.08) for n, (x, y) in pos.items()}
    # nx.draw_networkx_labels(
    #     model, label_pos,
    #     font_size=font_size,
    #     font_weight="bold",
    #     ax=ax
    # )
    # # Calculate arrow curvature based on number of edges
    # edge_count = len(model.edges())
    # base_rad = 0.25  # Default curvature

    # # For dense graphs, increase curvature variety
    # if edge_count > 0:
    #     # Create custom edge curvatures
    #     edge_rad = {}
    #     for i, edge in enumerate(model.edges()):
    #         # Alternate directions of curvature
    #         direction = 1 if i % 2 == 0 else -1
    #         # Scale curvature by edge density
    #         scale = 0.1 + 0.2 * (edge_count / (n_nodes * (n_nodes - 1)))
    #         # Vary curvature slightly for each edge
    #         edge_rad[edge] = direction * (base_rad + scale * (i % 3) / 3)

    #     # Custom connection style for each edge
    #     edge_styles = {}
    #     for edge in model.edges():
    #         rad = edge_rad[edge]
    #         edge_styles[edge] = f"arc3, rad={rad}"

    #     # Draw edges with custom styles
    #     for edge, style in edge_styles.items():
    #         nx.draw_networkx_edges(
    #             model, pos,
    #             edgelist=[edge],
    #             width=1.0,
    #             arrows=True,
    #             arrowsize=15,
    #             node_size=node_size,
    #             connectionstyle=style,
    #             ax=ax
    #         )
    # else:
    #     # If no custom styles, draw all edges the same way
    #     nx.draw_networkx_edges(
    #         model, pos,
    #         arrows=True,
    #         arrowsize=15,
    #         width=1.0,
    #         node_size=node_size,
    #         connectionstyle=f"arc3, rad={base_rad}",
    #         ax=ax
    #     )

    # plt.title(title, fontsize=max(14, 22 / np.log2(n_nodes + 2)))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    n_nodes = len(model.nodes())
    figsize_factor = max(1, np.log(n_nodes) / np.log(10) * 1.5)
    figsize = (8 * figsize_factor, 8 * figsize_factor)
    node_size = max(100, 1000 / np.log2(n_nodes + 2))
    font_size = max(8, 20 / np.log2(n_nodes + 2))

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.circular_layout(model)
    nx.draw_networkx(model, pos, with_labels=True, node_size=node_size,
                    node_color='skyblue', font_size=font_size, font_weight="bold",
                    arrows=True, arrowsize=15, connectionstyle="arc3, rad=0.1", ax=ax)
    plt.title(title)
    plt.savefig(save_path)
    model_logger.info(f"Saved Bayesian Network visualization to {save_path}")
    
def estimate_parameters(model: BayesianNetwork,
                        data: pd.DataFrame,
                        estimator_name: str = "MLE",
                        **kwargs) -> BayesianNetwork:
    model = model.copy()
    if estimator_name.upper() == "MLE":
        model.fit(data, estimator=MaximumLikelihoodEstimator, **kwargs)
        model.check_model()
        return model
    elif estimator_name.upper() == "BAYESIAN":
        model.fit(data, estimator=BayesianEstimator, **kwargs)
        # attempt validation; pgmpy BE usually provides CPDs for all nodes
        try:
            model.check_model()
        except Exception as e:
            # as a fallback: try to estimate missing CPDs by MLE one-by-one
            msg = str(e)
            if "No CPD associated with" in msg:
                missing = [n for n in model.nodes() if not model.get_cpds(n)]
                for node in missing:
                    cpd = MaximumLikelihoodEstimator(model, data).estimate_cpd(node)
                    model.add_cpds(cpd)
            model.check_model()
        return model
    else:
        raise ValueError("estimator_name must be 'MLE' or 'Bayesian'")

def build_expert_knowledge_for_dependents(df_wide: pd.DataFrame,
                                          role: str) -> Tuple[ExpertKnowledge, Optional[DAG]]:
    """
    the function builds ExpertKnowledge and an optional start DAG
    role ∈ {'children', 'others'}.
    """
    # the variable lists
    household_vars = [c for c in df_wide.columns if c in cons.HOUSEHOLD_COLS]
    head_vars = [c for c in df_wide.columns if c.startswith("Head_")]
    spouse_vars = [c for c in df_wide.columns if c.startswith("Spouse_")]
    if role == "children":
        dependent_vars = [c for c in df_wide.columns if c.startswith("Child_")]
    else:
        dependent_vars = [c for c in df_wide.columns if c.startswith("Other_")]

    # required core->dependent edges (dense fan-in is acceptable for structure search starting point)
    required_edges = [(p, d) for p in (household_vars + head_vars + spouse_vars) for d in dependent_vars]

    # forbidden edges: dependent->core; dependent<->dependent
    forbidden_edges = [(d, p) for p in (household_vars + head_vars + spouse_vars) for d in dependent_vars]
    forbidden_edges += [(a, b) for a in dependent_vars for b in dependent_vars if a != b]

    # build a starting DAG with required core->dependent edges (helps HillClimb)
    start_dag = DAG()
    start_dag.add_nodes_from(df_wide.columns)
    start_dag.add_edges_from(required_edges)

    ek = ExpertKnowledge(
        forbidden_edges=forbidden_edges
    )
    return ek, start_dag

def filter_bayesian_network(model: BayesianNetwork,
                            household_nodes: List[str],
                            head_nodes: List[str],
                            spouse_nodes: Optional[List[str]] = None,
                            child_nodes: Optional[List[str]] = None,
                            other_nodes: Optional[List[str]] = None) -> BayesianNetwork:
    spouse_nodes = spouse_nodes or []
    child_nodes = child_nodes or []
    other_nodes = other_nodes or []

    keep_edges = []

    # Household → Child / Other
    for s in household_nodes:
        for t in child_nodes:  keep_edges.append((s, t))
        for t in other_nodes:  keep_edges.append((s, t))
    # Head → Child / Other
    for s in head_nodes:
        for t in child_nodes:  keep_edges.append((s, t))
        for t in other_nodes:  keep_edges.append((s, t))
    # Spouse → Child / Other
    for s in spouse_nodes:
        for t in child_nodes:  keep_edges.append((s, t))
        for t in other_nodes:  keep_edges.append((s, t))

    filtered = model.copy()
    filtered.remove_edges_from([e for e in model.edges() if e not in keep_edges])
    return filtered

# ============================================
# Sampling
# ============================================
def sample_core(model: BayesianNetwork,
                n_real: int,
                multiplier: float,
                method: str) -> pd.DataFrame:
    size = int(round(n_real * multiplier))
    sampler = BayesianModelSampling(model)
    if method == "Forward":
        return sampler.forward_sample(size=size)
    elif method == "Gibbs":
        # Gibbs requires long burn-in/tuning; forward is preferred for discrete BNs
        return sampler.gibbs_sample(size=size)
    else:
        raise ValueError("CORE_SAMPLING_METHOD must be 'Forward' or 'Gibbs'")

def sample_dependents_per_household(role: str,
                                    bn_model: BayesianNetwork,
                                    core_row: pd.Series,
                                    household_vars: List[str],
                                    head_vars: List[str],
                                    spouse_vars: List[str],
                                    count_needed: int,
                                    method: str,
                                    lws_multiplier: int) -> pd.DataFrame:
    if count_needed <= 0:
        return pd.DataFrame()

    sampler = BayesianModelSampling(bn_model)

    # the evidence is the core + household part available in the dependent BN
    # Filter evidence_vars to only include columns present in core_row
    evidence_vars = [v for v in (household_vars + head_vars + spouse_vars) if v in bn_model.nodes() and v in core_row.index]
    evidence = [(var, core_row[var]) for var in evidence_vars]

    if method == "LikelihoodWeighted":
        n_draw = count_needed * lws_multiplier # or max(count_needed * lws_multiplier, 10)
        samples = sampler.likelihood_weighted_sample(evidence=evidence, size=n_draw)
        if samples.empty:
            return pd.DataFrame()
        weights = samples["_weight"].values
        if weights.sum() == 0:
            return pd.DataFrame()
        probs = weights / weights.sum()
        chosen = np.random.choice(samples.index, size=count_needed, replace=True, p=probs)
        out = samples.loc[chosen].copy()
        out["_household_index"] = core_row.name
        return out
    elif method == "Rejection":
        samples = sampler.rejection_sample(evidence=evidence, size=count_needed)
        if samples.empty:
            return pd.DataFrame()
        samples["_household_index"] = core_row.name
        return samples
    else:
        raise ValueError("DEPENDENT_SAMPLING_METHOD must be 'LikelihoodWeighted' or 'Rejection'")

def modify_edges_interactive(model: BayesianNetwork):
        """
        Interactively reverse or delete edges in a BN model.
        """
        modified_model = model.copy()
        print("Interactive Edge Modification (r=reverse, d=delete, k=keep)")
        for edge in list(model.edges()):
            action = input(f"Edge {edge[0]} -> {edge[1]} | Action (r/d/k): ").lower()
            if action == 'r':
                modified_model.remove_edge(*edge)
                modified_model.add_edge(edge[1], edge[0])
            elif action == 'd':
                modified_model.remove_edge(*edge)
        return modified_model