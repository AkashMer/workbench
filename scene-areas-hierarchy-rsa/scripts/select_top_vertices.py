"""Select the top-N most probable, spatially contiguous vertices for a
scene-area ROI label and write a new .label file.

Simply ranking all vertices by probability and taking the top N can produce a
spatially fragmented selection. Instead, this seeds the selection at the single 
highest-probability vertex and grows outward via the surface mesh's own vertex 
adjacency, always expanding to whichever unvisited neighbor has the highest 
probability next, until N vertices are collected.
"""
# Import necessary libraries
import argparse
import heapq
from pathlib import Path

import nibabel as nib
import numpy as np

# Builds a per-vertex neighbor list from the faces
def build_adjacency(faces, n_vertices):
    # One set of neighbor indices per vertex
    adjacency = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        # Each triangle face connects all 3 of its vertices pairwise
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency

# Expands outward from the highest-probability vertex, picking the highest-probability
def grow_from_peak(nonzero_indices, nonzero_probs, adjacency, n):

    # unvisited neighbor at each step, until n vertices are selected
    prob_by_vertex = dict(zip(nonzero_indices, nonzero_probs))
    peak_vertex = nonzero_indices[np.argmax(nonzero_probs)]

    selected = set()
    # Max-heap via negated probability, seeded with the peak vertex
    frontier = [(-prob_by_vertex[peak_vertex], peak_vertex)]
    visited = {peak_vertex}

    while frontier and len(selected) < n:
        # Pop the highest-probability vertex still on the frontier
        neg_prob, vertex = heapq.heappop(frontier)
        selected.add(vertex)
        for neighbor in adjacency[vertex]:
            # Only grow into unvisited vertices that are part of this region
            if neighbor in visited or neighbor not in prob_by_vertex:
                continue
            visited.add(neighbor)
            heapq.heappush(frontier, (-prob_by_vertex[neighbor], neighbor))

    return np.array(sorted(selected))

# Loads the region's per-vertex probabilities, selects a contiguous top-n set, writes a .label file
def select_top_vertices(prob_path: Path, surf_path: Path, out_path: Path, n: int = 800):

    if prob_path.suffix == ".gii":
        # Run this block if the file is in Gifti format
        prob_data = nib.load(prob_path).darrays[0].data
        # Drop any zeros so the areas outside the brain are not included
        nonzero_indices = np.nonzero(prob_data)[0]
        nonzero_probs = prob_data[nonzero_indices]
    else:
        # Run this block for FreeSurfer label files
        nonzero_indices, nonzero_probs = nib.freesurfer.io.read_label(prob_path, read_scalars=True)

    # Get the cooridinates of the vetices
    coords, faces = nib.freesurfer.io.read_geometry(surf_path)
    adjacency = build_adjacency(faces, coords.shape[0])

    # Find the highest probability vertex and walk outwards from there
    top_n = min(n, nonzero_indices.size)
    indices = grow_from_peak(nonzero_indices, nonzero_probs, adjacency, top_n)

    with open(out_path, "w") as f:
        f.write(f"#!ascii label, top {top_n} contiguous vertices\n{indices.size}\n")
        for idx in indices:
            x, y, z = coords[idx]
            f.write(f"{idx} {x:.3f} {y:.3f} {z:.3f} 1.0\n")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-path", type=Path, required=True)
    parser.add_argument("--surf-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--n", type=int, default=800)
    args = parser.parse_args()

    select_top_vertices(args.prob_path, args.surf_path, args.out_path, args.n)
