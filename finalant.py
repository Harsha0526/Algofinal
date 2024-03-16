import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tsplib95

def compute_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.sqrt(((cities[np.newaxis, :, :] - cities[:, np.newaxis, :]) ** 2).sum(axis=2))
    dist_matrix += np.eye(n) * 1e-9  # Add a small value to diagonal to avoid division by zero
    return dist_matrix

def visualize_tour(cities, tour, distances, title=""):
    plt.figure(figsize=(10, 7))
    plt.title(title)
    n = len(cities)
    for i in range(-1, n - 1):
        plt.plot([cities[tour[i], 0], cities[tour[i + 1], 0]], [cities[tour[i], 1], cities[tour[i + 1], 1]], 'b-')
    plt.scatter(cities[:, 0], cities[:, 1], c='red')
    for i in range(len(tour)):
        city1 = tour[i]
        city2 = tour[(i + 1) % len(tour)]  # Adjust index to loop around to the start
        plt.text((cities[city1, 0] + cities[city2, 0]) / 2, (cities[city1, 1] + cities[city2, 1]) / 2, f'{distances[i]:.2f}', color='green')
    plt.show()

# Ant Colony Optimization implementation with 2-opt improvement
def ant_colony_optimization_2opt(cities, distance_matrix, alpha, beta, num_ants=20, num_iterations=100, rho=0.5):
    num_cities = len(cities)
    pheromone = np.ones((num_cities, num_cities)) / num_cities
    best_tour = None
    best_distance = np.inf
   
    for iteration in range(num_iterations):
        all_tours = []
        for ant in range(num_ants):
            tour = construct_initial_tour(num_cities, distance_matrix, pheromone, alpha, beta)
            tour = apply_two_opt(tour, distance_matrix)  # Apply 2-opt heuristic
            tour_distance = calculate_tour_distance(tour, distance_matrix)
            all_tours.append((tour, tour_distance))
            if tour_distance < best_distance:
                best_tour = tour[:]
                best_distance = tour_distance
       
        # Update pheromone levels
        pheromone *= (1 - rho)  # Evaporation
        for tour, tour_distance in all_tours:
            for i in range(len(tour) - 1):
                pheromone[tour[i], tour[i + 1]] += 1.0 / tour_distance  # Pheromone trail deposit
       
    return best_tour, best_distance

def construct_initial_tour(num_cities, distance_matrix, pheromone, alpha, beta):
    tour = [np.random.randint(num_cities)]
    visited = set(tour)
    while len(visited) < num_cities:
        probabilities = ((pheromone[tour[-1]] ** alpha) * ((1.0 / distance_matrix[tour[-1]]) ** beta))
        probabilities[list(visited)] = 0  # Mask out visited cities
        probabilities /= probabilities.sum()
        next_city = np.random.choice(np.arange(num_cities), p=probabilities)
        tour.append(next_city)
        visited.add(next_city)
    return tour

def apply_two_opt(tour, distance_matrix):
    num_cities = len(tour)
    best_distance = calculate_tour_distance(tour, distance_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue  # changes nothing, skip then
                new_tour = tour[:]
                new_tour[i:j] = reversed(tour[i:j])  # two-opt swap
                new_distance = calculate_tour_distance(new_tour, distance_matrix)
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
        if improved:
            break
    return tour

def calculate_tour_distance(tour, distance_matrix):
    return np.sum(distance_matrix[np.array(tour), np.roll(np.array(tour), -1)])

# Function to read coordinates from a TSP file using tsplib95
def read_tsp_coordinates(file_path):
    tsp_data = tsplib95.load(file_path)
    coordinates = np.array([tsp_data.node_coords[i+1] for i in range(tsp_data.dimension)])
    return coordinates

# Merged comparison function with visualization
def compare_algorithms_with_visualization(file_path, optimal_distance=None):
    coordinates = read_tsp_coordinates(file_path)
    instance_name = os.path.basename(file_path)
    cities = np.array(coordinates)
    dist_matrix = compute_distance_matrix(cities)
   
    alpha_val = 1
    beta_val = 2
   
    start_time = time.time()
    tour_aco, aco_distance = ant_colony_optimization_2opt(cities, dist_matrix, alpha_val, beta_val)
    aco_duration = time.time() - start_time
    
    print("Instance Name:", instance_name)
    print("Time Taken:", aco_duration, "seconds")
    print("ACO Distance:", aco_distance)
    
    if optimal_distance is not None:
        error_percentage = ((aco_distance - optimal_distance) / optimal_distance) * 100
        print("best:", optimal_distance)
        print("Error Percentage:", error_percentage, "%")
    else:
        print("Optimal Distance: Not Provided")
    
    print("Distances between consecutive points:")
    total_distance = 0
    distances = []
    for i in range(len(tour_aco)):
        city1 = tour_aco[i]
        city2 = tour_aco[(i + 1) % len(tour_aco)]  # Adjust index to loop around to the start
        distance = dist_matrix[city1, city2]
        total_distance += distance
        distances.append(distance)
    
    visualize_tour(cities, tour_aco, distances, f"ACO with 2-opt for {instance_name}")


# Example Usage
file_path = '/Users/harsha/Downloads/pr124.tsp'
compare_algorithms_with_visualization(file_path, 49135)
