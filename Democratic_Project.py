import numpy as np
import random
import math
import time
import heapq
import statistics
import matplotlib.pyplot as plt
import itertools

def labor():
    constant_vector = np.random.randint(1, 11, size=250)
    for i in range(0, 250, 25):
        constant_vector[i] = np.random.randint(1, 11)
    return constant_vector

def machine_time(sequence_length=250, num_sequences=200):
    sequences = []
    mu = random.randint(30, 60)
    tau = random.randint(10, 25)
    while len(sequences) < num_sequences:
        sequence = []
        consecutive_ones = 0
        while len(sequence) < sequence_length:
            if consecutive_ones < mu:
                if random.random() < 0.5:
                    sequence.append(1)
                    consecutive_ones += 1
                else:
                    sequence.append(0)
            else:
                for _ in range(tau):
                    sequence.append(0)
                    if len(sequence) >= sequence_length:
                        break
                consecutive_ones = 0  
            
        sequences.append(sequence[:sequence_length])  

    return sequences

    
def bookFunction(i):
    FunctionsBook = {}
    FunctionsBook['Power_cons'] = [np.random.uniform(10, 60, size=250).tolist() for _ in range(i)]  # Convert to list
    FunctionsBook['Ideal_cost'] = [np.full(250, 0).tolist()]  # Convert to list
    FunctionsBook['Ideal_labor'] = [labor().tolist()]  # Convert to list
    # Ensure other function values are generated as lists or arrays
    FunctionsBook['Machine_1'] = machine_time()
    FunctionsBook['Machine_2'] = machine_time()
    FunctionsBook['Machine_3'] = machine_time()
    FunctionsBook['Machine_4'] = machine_time()
    FunctionsBook['Machine_5'] = machine_time()
    FunctionsBook['Ranking'] = [list(range(1, 251))]
    return FunctionsBook


def l1_distance(vec1, vec2):
    return sum(abs(vec1[i] - vec2[i]) for i in range(len(vec1)))

def generate_positions(my_list):
    copy=my_list
    indexed_list = list(enumerate(my_list))
    flag=1
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    for i in range(len(my_list)):
        index = sorted_list[i][0]
        copy[index]=flag
        flag+=1
    return copy

def swap_distance(vector1,vector2):
    swaps = 0
    vector1pos=generate_positions(vector1)
    for i in range(len(vector1)):
        if vector1[i] != vector2[i]:
            index = vector2.index(vector1pos[i])
            cost = abs(vector1[i] - vector2[i]) 
            swaps += cost
            vector1[i], vector1[index] = vector1[index], vector1[i]
            vector1pos[i], vector1pos[index] = vector1pos[index], vector1pos[i]
    return swaps

def samples(i):
    instance=random.sample(bigJobs,i)
    return instance

def Scoring(solution, functions, distance, X):
    scores = []
    for func in functions:
        if distance == 'l1':
            scores.append(l1_distance(solution, func))
        else:
            scores.append(swap_distance(solution, func))
    return min(scores)

def get_values_by_key(tuples, key):
    values = []
    for t in tuples:
        if key in t:
            values.append(t[key])
    return values

def cost(permutation, X, FunctionsBook):
    cost = 0
    t = random.uniform(2, 5)
    z = random.uniform(1, 2)
    weights = [
        t, t, t, z,
        z, z, z, z, random.uniform(4, 5)
    ]
    permutationBook = {
        'Power_cons': get_values_by_key(permutation, 'power_consumption'),
        'Ideal_cost': get_values_by_key(permutation, 'job_cost'),
        'Ideal_labor': get_values_by_key(permutation, 'labor_requirement'),
        'Machine_1': get_values_by_key(permutation, 'machine_1'),
        'Machine_2': get_values_by_key(permutation, 'machine_2'),
        'Machine_3': get_values_by_key(permutation, 'machine_3'),
        'Machine_4': get_values_by_key(permutation, 'machine_4'),
        'Machine_5': get_values_by_key(permutation, 'machine_5'),
        'Ranking': get_values_by_key(permutation, 'job_ranking')
    }
    j = 0
    for i in permutationBook.keys():
        if permutationBook[i]!='Ranking':
            cost += weights[j] * Scoring(permutationBook[i], FunctionsBook[i], 'l1', X)
        else:
            cost += weights[j] * Scoring(permutationBook[i], FunctionsBook[i], 'swap', X)
        j += 1
    return cost


def zRandom(z, sol, i, FunctionsBook):
    scores = []
    for _ in range(z):
        x = cost(random.sample(sol, i), i, FunctionsBook)
        scores.append(x)
    return min(scores)

def greedy_algorithm(size, completePlay, FunctionsBook, ran):
    # Use a priority queue (min-heap)
    heap = []
    
    # Compute initial costs once
    initial_costs = {}
    for element in completePlay:
        element_frozen = frozenset(element.items())  # Hashable type
        initial_costs[element_frozen] = cost([element], size, FunctionsBook)
        heapq.heappush(heap, (initial_costs[element_frozen], element))
    
    # Initialize
    permutation = []
    selected_jobs = set()
    
    # Extract the best job first
    min_cost, best_element = heapq.heappop(heap)
    permutation.append(best_element)
    selected_jobs.add(frozenset(best_element.items()))

    # Maintain a running total cost instead of recomputing
    running_cost = min_cost / ran

    for _ in range(size - 1):  # We already selected 1 job
        if not heap:
            break
        
        # Get the next lowest cost job that is not in selected_jobs
        while heap:
            min_cost, best_element = heapq.heappop(heap)
            best_element_frozen = frozenset(best_element.items())
            if best_element_frozen not in selected_jobs:
                break  

        # Add the best element to the permutation
        permutation.append(best_element)
        selected_jobs.add(best_element_frozen)  # Mark as used
        
        # Incrementally update the cost instead of recomputing fully
        running_cost += min_cost / ran

    return running_cost,permutation


def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)

def simulated_annealingReg(size, completePlay, FunctionsBook, ran, per):
    cost_history = []
    temperature = 1000
    cooling_rate = 0.003
    permutation = per  # Start with the provided Greedy solution
    minimal_permutation = per  # Track the best solution found
    old_cost = cost(permutation, size, FunctionsBook)  # Cost of the initial solution
    minimal_cost = old_cost  # Initialize minimal cost with the initial cost
    start_time = time.time()  # Record the starting time
    last_recorded_time = start_time  # Track last measurement
    end_time = start_time + 600  # Run for 10 minutes
    improvement_counter = 0
    total_counter = 0
    target_acceptance_rate = 0.5  # Set the target acceptance rate to 50%
    counter = 0

    # **Initial measurement before the process starts**
    cost_history.append(minimal_cost / ran)

    while time.time() < end_time:
        # Generate a new permutation by randomly modifying the current one
        new_permutation = permutation[:]
        index1 = random.randint(0, size - 1)
        t = new_permutation[index1]

        # Find a random replacement for the job at `index1`
        track = random.sample(completePlay, 1)
        while track == t:
            track = random.sample(completePlay, 1)

        # Replace or swap jobs in the new permutation
        if track in permutation:
            index2 = new_permutation.index(track)
            new_permutation.remove(track)
            new_permutation.remove(t)
            new_permutation.insert(index1, track)
            new_permutation.insert(index2, t)
        else:
            new_permutation.remove(t)
            new_permutation.insert(index1, track)

        # Calculate the cost of the new permutation
        new_cost = cost(new_permutation, len(new_permutation), FunctionsBook)

        # Determine if the new solution is accepted
        acceptance_prob = acceptance_probability(old_cost, new_cost, temperature)
        if random.random() < acceptance_prob:
            permutation = new_permutation  # Accept the new solution
            old_cost = new_cost  # Update the old cost
            improvement_counter += 1

        # Update the minimal cost and best solution if improvement is found
        if new_cost < minimal_cost:
            minimal_cost = new_cost
            minimal_permutation = new_permutation

        # Update total iterations and acceptance rate
        total_counter += 1
        acceptance_rate = improvement_counter / total_counter

        # Adjust cooling rate based on acceptance rate
        if acceptance_rate > target_acceptance_rate:
            cooling_rate *= 1.05  # Increase cooling rate to converge faster
        else:
            cooling_rate *= 0.95  # Decrease cooling rate to explore more of the search space

        # Update temperature
        temperature *= 1 - cooling_rate
        counter += 1

        # Reset the permutation every 6000 iterations, but retain the best solution
        if counter == 5000:
            permutation = minimal_permutation[:]  # Restart from the best solution
            old_cost = minimal_cost
            counter = 0

        # Record the minimal cost every 60 seconds
        current_time = time.time()
        if current_time - last_recorded_time >= 60:
            cost_history.append(minimal_cost / ran)
            last_recorded_time = current_time  # Reset last recorded time

    # **Final measurement at the last second**
    cost_history.append(minimal_cost / ran)

    return cost_history



def adaptive_crossover_prob(generation):
    # Set the initial crossover probability
    initial_crossover_prob = 0.85
    # Set the minimum crossover probability
    min_crossover_prob = 0.5
    # Set the maximum generation number
    max_generation = 1000
    
    # Calculate the current crossover probability
    crossover_prob = max(initial_crossover_prob - (generation / max_generation) *
                        (initial_crossover_prob - min_crossover_prob), min_crossover_prob)
    return crossover_prob

def crossover(permutation1, permutation2, size):
    crossover_point = random.randint(1, size-1)
    # Generate the new permutation by combining the two parent permutations
    new_permutation = permutation1[:crossover_point]
    for i in range(crossover_point, size):
        item = permutation2[i]
        if item not in new_permutation:
            new_permutation.append(item)
        else:
            new_item = permutation1[i]
            new_permutation.append(new_item)
    return new_permutation

def mutate(permutation, original_size, completePlay):
    sam = random.sample(completePlay, 1)
    index = random.randint(0, len(permutation) - 1)
    while sam in permutation:
        sam = random.sample(completePlay, 1)
    permutation.remove(permutation[index])
    permutation.insert(index, sam)
    return permutation

def genetic_algorithm(completePlay, size, FunctionsBook, ran, per):
    cost_history = []
    population_size = 100  # Start small, let it adjust dynamically
    max_population_growth = 1.2  # Population expands at most 20%
    min_population_reduction = 0.8  # Population shrinks to 80% on stagnation
    max_stagnant_generations = 1000 # Restart threshold for stagnation
    stagnant_generations = 0

    # Initialize Greedy solution
    minimal_cost = cost(per, size, FunctionsBook)  # Start with Greedy cost
    minimal_permutation = per  # Track the best solution

    # Initialize population
    def initialize_population(per):
        population = [per]  # Start with the Greedy solution
        for _ in range(population_size - 1):
            population.append(random.sample(completePlay, size))  # Add random permutations
        return population

    population = initialize_population(per)

    start_time = time.time()  # Record starting time
    last_recorded_time = start_time  # Track the last recorded measurement
    end_time = start_time + 600  # Run for 10 minutes
    generation = 0

    # **Initial measurement before the process starts**
    cost_history.append(minimal_cost / ran)

    while time.time() < end_time:
        generation += 1

        # Compute population costs
        population_cost = [cost(perm, size, FunctionsBook) for perm in population]

        # Sort population by cost (lower is better)
        population_sorted = sorted(zip(population, population_cost), key=lambda x: x[1])
        population, population_cost = zip(*population_sorted)
        population = list(population)
        population_cost = list(population_cost)

        # Update best solution
        if population_cost[0] < minimal_cost:
            minimal_cost = population_cost[0]
            minimal_permutation = population[0]
            stagnant_generations = 0  # Reset stagnation counter
        else:
            stagnant_generations += 1

        # Reset population if stagnation occurs
        if stagnant_generations >= max_stagnant_generations:
            population = initialize_population(minimal_permutation)  # Restart from the best solution
            stagnant_generations = 0
            continue

        # Record minimal cost every 60 seconds
        current_time = time.time()
        if current_time - last_recorded_time >= 60:
            cost_history.append(minimal_cost / ran)
            last_recorded_time = current_time  # Reset last recorded time

        # Adjust population size dynamically based on diversity
        diversity = statistics.stdev(population_cost)
        if diversity < 0.1 * minimal_cost:  # Shrink if diversity is too low
            population_size = max(int(population_size * min_population_reduction), 50)
        elif generation % 50 == 0:  # Expand periodically
            population_size = min(int(population_size * max_population_growth), 1000)

        # Generate new population with crossover/mutation
        new_population = []
        while len(new_population) < population_size:
            crossover_prob = adaptive_crossover_prob(generation)
            if random.random() < crossover_prob:
                parent1, parent2 = random.sample(range(len(population)), 2)
                new_perm = crossover(population[parent1], population[parent2], size)
            else:
                parent = random.randint(0, len(population) - 1)
                new_perm = mutate(population[parent], size, completePlay)
            new_population.append(new_perm)

        population = new_population

    # **Final measurement at the last second**
    cost_history.append(minimal_cost / ran)

    return cost_history




# Total number of jobs and jobs to rank
num_jobs = 5000
num_jobs_to_rank = 250
num_agents = 20

# Set the total sum of job rankings
total_sum_jobs =  627500

# Generate job rankings ensuring the total sum constraint and capping each job's score
job_rankings = np.zeros(num_jobs)

# Distribute the total sum across jobs, capping the score to ensure it stays within the range
for i in range(num_jobs_to_rank):
    # Calculate the sum for each job
    job_sum = min(total_sum_jobs, num_agents * (num_jobs_to_rank - i))
    
    # Assign the calculated sum to the job
    job_rankings[i] = job_sum
    total_sum_jobs -= job_sum

# Shuffle the job rankings
np.random.shuffle(job_rankings)

# Create an array of dictionaries representing jobs with their attributes and values
bigJobs = []
for i in range(num_jobs):
    num_machines = 5
    activated_machine = random.randint(1, num_machines)
    job_attributes = {
        'job_cost': np.random.uniform(100, 1000),
        'power_consumption': np.random.uniform(10, 100),
        'labor_requirement': np.random.randint(1, 10),
        **{f"machine_{j}": 1 if j == activated_machine else 0 for j in range(1, num_machines + 1)},
        'job_ranking': min(job_rankings[i], num_agents * num_jobs_to_rank)  # Cap the job ranking score
    }
    bigJobs.append(job_attributes)
# Graphing
gen = []
greed = []
simReg = []
instances = []
zRand = []
FunctionsBook = bookFunction(250)

for i in range(10):
    instances.append(samples(700))

for instance in instances:
    ran = zRandom(100, instance, 250, FunctionsBook)
    minimal_cost,permutation = greedy_algorithm(250, instance, FunctionsBook, ran)
    costArr2 = genetic_algorithm(instance, 250, FunctionsBook, ran,permutation)
    costArr3 = simulated_annealingReg(250, instance, FunctionsBook, ran,permutation)
    gen.append(costArr2)
    greed.append(minimal_cost)
    simReg.append(costArr3)
    zRand.append(ran)

timeArr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gen_mean = []
sim_mean = []
simReg_mean = []
mean_greed = []

for a in timeArr:
    total2 = 0  # Simulated costs
    total3 = 0  # Genetic costs
    for b in range(10):  # Iterate over instances
        total2 += simReg[b][a]  # Simulated at time `a`
        total3 += gen[b][a]  # Genetic at time `a`
    gen_mean.append(total3 / 10)
    simReg_mean.append(total2 / 10)
    mean_greed.append(np.mean(greed))

# Plotting
plt.plot(timeArr, gen_mean, label='Genetic', color='red', linestyle='dotted', markersize=1)
plt.plot(timeArr, simReg_mean, label='Simulated', color='green', linestyle='dashed', markersize=1)
plt.plot(timeArr, mean_greed, label='Greedy', color='blue', markersize=1)
plt.xlabel('Minutes')
plt.ylabel('Normalized Average Cost')
plt.legend()
plt.show()
