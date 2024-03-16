import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import time

def read_coordinates_from_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].isdigit():
                coordinates.append((float(parts[1]), float(parts[2])))
    return np.array(coordinates)

def create_complete_graph(coordinates):
    G = nx.Graph()
    for i, coord1 in enumerate(coordinates):
        for j, coord2 in enumerate(coordinates):
            if i != j:
                distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
                G.add_edge(i, j, weight=distance)
    return G

def christofides_tsp(G):
    T = nx.minimum_spanning_tree(G)
    odd_vertices = [v for v, degree in T.degree() if degree % 2 != 0]
    M = nx.min_weight_matching(G.subgraph(odd_vertices).copy())
    H = nx.MultiGraph(T.edges())
    H.add_edges_from(M)
    euler_circuit = list(nx.eulerian_circuit(H, 0))
    tsp_tour = [euler_circuit[0][0]]
    for o, d in euler_circuit:
        if d not in tsp_tour:
            tsp_tour.append(d)
    return tsp_tour

def calculate_distance_tsp(G, order):
    total_distance = 0
    for i in range(len(order) - 1):
        total_distance += G[order[i]][order[i + 1]]['weight']
    total_distance += G[order[-1]][order[0]]['weight']  # Return to starting city
    return total_distance

# Read coordinates from file
file_path = r'/Users/harsha/Downloads/berlin52.tsp'  # Specify the path to your file
coordinates = read_coordinates_from_file(file_path)

# Create a complete graph based on the coordinates
G = create_complete_graph(coordinates)

print("Number of Nodes:", G.number_of_nodes())
print("Number of Edges:", G.number_of_edges())

# Calculate tour using Christofides
start_time = time.time()
tour = christofides_tsp(G.copy())
execution_time = time.time() - start_time
tour_distance = calculate_distance_tsp(G, tour)

print("Christofides Tour:", len(tour))
print("Tour Distance:", tour_distance)
print("Execution Time:", execution_time, "seconds")

# Plot only the tour path
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)

# Draw edges without arrows
nx.draw_networkx_edges(G, pos, edgelist=[(tour[i], tour[i + 1]) for i in range(len(tour) - 1)] + [(tour[-1], tour[0])],
                       width=2, alpha=0.5, edge_color='b', arrows=False)

# Add arrows manually
for i in range(len(tour) - 1):
    plt.annotate("", xy=pos[tour[i + 1]], xytext=pos[tour[i]],
                 arrowprops=dict(arrowstyle="->", lw=2, alpha=0.5, color='b'))

# Add arrow for the last edge
plt.annotate("", xy=pos[tour[0]], xytext=pos[tour[-1]],
             arrowprops=dict(arrowstyle="->", lw=2, alpha=0.5, color='b'))

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')

plt.title("Tour Path generated by Christofides Algorithm")
plt.show()
def calculate_error_percentage(result, opt):
    error_percentage = ((result - opt) / opt) * 100
    return error_percentage

# Example usage:
optimal_distance = 675  # Replace this value with the optimal solution value
error_percent = calculate_error_percentage(tour_distance, optimal_distance)
print("Error Percentage:", error_percent, "%")