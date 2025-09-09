import json
import argparse
from pathlib import Path
from itertools import cycle
from typing import Iterator

import graphviz

# A more vibrant and accessible color palette
COLORS = cycle([
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
])

def create_graph(data: dict, do_cluster: bool) -> graphviz.Digraph:
    """Creates a directed graph from the JSON data structure."""
    
    # Use Digraph for directed graphs.
    dot = graphviz.Digraph(comment='Data Flow Graph')
    dot.attr(splines='true', overlap='false', concentrate='true')
    
    node_colors = {}

    # --- 1. Add Nodes ---
    # With clustering, we add nodes inside their respective subgraphs.
    # Without clustering, we add them to the main graph.
    if not do_cluster:
        for node in data['nodes']:
            stage_name = node['id']
            color = next(COLORS)
            node_colors[stage_name] = color
            dot.node(stage_name, color=color, style='filled', fillcolor=f'{color}20') # Light fill
            
    # --- 2. Add Package Clusters (if enabled) ---
    else:
        subgraphs = {}
        # A helper to iterate through package names like 'a.b.c' -> 'a', 'a.b'
        def package_iterator(class_name: str) -> Iterator[str]:
            i = class_name.find('.')
            while i > 0:
                yield class_name[0:i]
                i = class_name.find('.', i + 1)

        for node in data['nodes']:
            stage_name = node['id']
            color = next(COLORS)
            node_colors[stage_name] = color
            
            # Start with the main graph as the parent
            parent_graph = dot 
            for package in package_iterator(stage_name):
                # If the subgraph doesn't exist, create it inside its parent
                if package not in subgraphs:
                    # Name must start with 'cluster_' for a visible box
                    subgraph = graphviz.Digraph(name=f'cluster_{package}')
                    subgraph.attr(label=package, style='filled', color='lightgrey')
                    parent_graph.subgraph(subgraph)
                    subgraphs[package] = subgraph
                # The new parent is the subgraph we just found or created
                parent_graph = subgraphs[package]

            # Add the final node inside the deepest parent subgraph
            parent_graph.node(stage_name, color=color, style='filled', fillcolor=f'{color}20')

    # --- 3. Add Edges ---
    for link in data['links']:
        source = link['source']
        target = link['target']
        # Use the stored color of the source node for the edge
        edge_color = node_colors.get(source, 'black')
        dot.edge(source, target, color=edge_color)

    return dot

def main():
    """Main function to parse arguments and generate the graph."""
    parser = argparse.ArgumentParser(description="Generate a data flow graph from a JSON file.")
    parser.add_argument(
        '-j', '--json-input',
        type=Path,
        required=True,
        help="Input JSON file to plot."
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('data_flow.png'),
        help="Write the flow graph to OUTPUT_FILE (e.g., data_flow.png)."
    )
    parser.add_argument(
        '-g', '--cluster',
        action='store_true',
        help="Group nodes into package clusters."
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format from file extension
    output_format = args.output.suffix.lstrip('.')
    if not output_format:
        raise ValueError("Output file must have an extension (e.g., .png, .svg, .pdf).")

    try:
        with args.json_input.open('r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.json_input}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {args.json_input}")
        return

    # Analyze and visualize
    graph = create_graph(data, args.cluster)
    
    # The render method saves the file. The filename is the first arg.
    output_filename_stem = args.output.with_suffix('')
    graph.render(output_filename_stem, format=output_format, view=False, cleanup=True)
    
    print(f"Graph successfully generated at: {args.output}")

if __name__ == '__main__':
    main()