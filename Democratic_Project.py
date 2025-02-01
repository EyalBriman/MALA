import numpy as np
import random
import math
import time
import heapq
import statistics
import matplotlib.pyplot as plt
import itertools
def zRandom(z, sol, i, FunctionsBook, weights):
    """
    Runs z random trials, sampling 'i' jobs from 'sol' each time, and computes their cost.
    Returns both the minimum cost and the corresponding best permutation.
    """
    min_score = float('inf')  # Initialize min score
    best_permutation = None   # Store best permutation

    for _ in range(z):
        permutation = random.sample(sol, i)  # Sample 'i' jobs
        score = cost(permutation, i, FunctionsBook, weights)  # Compute cost
        
        if score < min_score:  # Update minimum score and best permutation
            min_score = score
            best_permutation = permutation

    return min_score, best_permutation

def greedy_algorithm(size, completePlay, FunctionsBook, ran, weights):
    # Use a priority queue (min-heap)
    heap = []
    
    # Compute initial costs for each job
    for index, element in enumerate(completePlay):
        initial_cost = cost([element], size, FunctionsBook, weights)  # Compute cost if alone
        heapq.heappush(heap, (initial_cost, index, element))  # Store tuple (cost, index, element)

    # Initialize with best first job
    best_cost, _, best_element = heapq.heappop(heap)
    permutation = [best_element]
    selected_jobs = {frozenset(best_element.items())}

    for _ in range(size - 1):  # We already selected 1 job
        if not heap:
            break
        
        best_candidate = None
        best_marginal_cost = float('inf')

        # Check every job and find the one with the lowest **marginal** cost
        for _ in range(len(heap)):
            candidate_cost, _, candidate = heapq.heappop(heap)  # Extract job and cost
            candidate_frozen = frozenset(candidate.items())

            if candidate_frozen in selected_jobs:
                continue  # Skip already selected jobs

            # Compute marginal cost if we add this job to the permutation
            marginal_cost = cost(permutation + [candidate], size, FunctionsBook, weights)

            if marginal_cost < best_marginal_cost:
                best_marginal_cost = marginal_cost
                best_candidate = candidate

        if best_candidate:
            # Add best candidate to sequence
            permutation.append(best_candidate)
            selected_jobs.add(frozenset(best_candidate.items()))

    return permutation  # Return only the permutation



def acceptance_probability(old_cost, new_cost, temperature):
    """
    Computes acceptance probability for simulated annealing,
    preventing overflow errors when computing exp().
    """
    if new_cost < old_cost:
        return 1.0
    else:
        exponent = (old_cost - new_cost) / max(temperature, 1e-10)  # Avoid division by zero
        exponent = min(exponent, 500)  # Prevent overflow
        return np.exp(exponent)

def simulated_annealingReg(size, completePlay, FunctionsBook, ran, per, mcost, weights):
    cost_history = []
    
    # **Adaptive Initial Temperature**
    temperature = np.mean([mcost]) * 10  # Increase temperature for better exploration
    initial_temperature = temperature  # Store the initial value for random restarts
    cooling_rate = 0.995  # Ensure it **gradually decreases**
    stagnation_threshold = 5000  # Restart if no improvement for 2000 iterations
    stagnation_counter = 0  # Track stagnation count

    permutation = per[:]  # **Start with `zRandom` solution**
    minimal_permutation = per[:]
    old_cost = ran  # **Start with `zRandom` cost**
    minimal_cost = old_cost  # Best cost seen so far

    start_time = time.time()
    last_recorded_time = start_time
    end_time = start_time + 600  # Run for 10 minutes
    improvement_counter = 0
    total_counter = 0
    target_acceptance_rate = 0.5
    counter = 0

    # **🟢 Initial Measurement Before the Process Starts**
    cost_history.append(minimal_cost / ran)
    #print(f"🟢 Initial Cost (zRandom Solution): {minimal_cost / ran:.6f}")

    while time.time() < end_time:
        # **🔹 Select a job to swap intelligently**
        new_permutation = permutation[:]

        # **Swap Two Jobs Instead of One**
        index1, index2 = random.sample(range(size), 2)  # Pick 2 random indices
        new_permutation[index1], new_permutation[index2] = new_permutation[index2], new_permutation[index1]

        # **🔹 Calculate new cost**
        new_cost = cost(new_permutation, len(new_permutation), FunctionsBook, weights)

        # **🔹 Print Debug Info**
        #print(f"🔍 Iteration {counter}: Old Cost: {old_cost:.6f}, New Cost: {new_cost:.6f}, Temp: {temperature:.4f}")

        # **🔹 Compute Acceptance Probability**
        acceptance_prob = acceptance_probability(old_cost, new_cost, temperature)
        if random.random() < acceptance_prob:
            permutation = new_permutation[:]
            old_cost = new_cost
            improvement_counter += 1
            if new_cost < minimal_cost:
                minimal_cost = new_cost
                minimal_permutation = new_permutation[:]
                #print(f"✅ New Best Solution Found: {minimal_cost / ran:.6f}")

        else:
            stagnation_counter += 1  # Increment stagnation counter

        # **🔹 Check for Random Restart**
        if stagnation_counter >= stagnation_threshold:
            #print(f"🔄 Random Restart at Iteration {total_counter} - No improvement for {stagnation_threshold} steps.")
            permutation = minimal_permutation[:]  # **Restart from best found so far**
            random.shuffle(permutation)  # Shuffle slightly for randomness
            old_cost = cost(permutation, len(permutation), FunctionsBook, weights)
            temperature = initial_temperature  # 🔥 Reset temperature properly
            stagnation_counter = 0  # Reset stagnation counter

        # **🔹 Compute Acceptance Rate**
        total_counter += 1
        acceptance_rate = improvement_counter / total_counter

        # **🔹 Adjust Cooling Rate Dynamically**
        if acceptance_rate > target_acceptance_rate:
            cooling_rate = max(0.990, cooling_rate * 0.99)  # Keep it **decreasing**
        else:
            cooling_rate = min(0.999, cooling_rate * 1.01)  # Prevent too slow cooling

        # **🔹 Update Temperature Safely**
        temperature *= cooling_rate  # Reduce temperature slower

        counter += 1

        # **🔹 Record minimal cost every 60 seconds**
        current_time = time.time()
        if current_time - last_recorded_time >= 60:
            cost_history.append(minimal_cost / ran)
            last_recorded_time = current_time
            #print(f"🕒 Time {int(current_time - start_time)//60} min: Best Cost: {minimal_cost / ran:.6f}")

    # **🔹 Final Cost Measurement**
    cost_history.append(minimal_cost / ran)
    #print(f"🏁 Final Cost: {minimal_cost / ran:.6f}")

    return cost_history


def adaptive_crossover_prob(generation, max_generation=1000, initial_prob=0.85, min_prob=0.5):
    """Adaptive crossover probability decreases as generations progress."""
    return max(initial_prob - (generation / max_generation) * (initial_prob - min_prob), min_prob)

def crossover(parent1, parent2, size):
    """Performs order-preserving crossover to ensure diversity."""
    crossover_point = random.randint(1, size - 1)
    new_permutation = parent1[:crossover_point]  
    for job in parent2:
        if job not in new_permutation:
            new_permutation.append(job)
    return new_permutation

def mutate(permutation, size, completePlay, mutation_rate=0.1):
    """Mutates the permutation by swapping or replacing jobs."""
    if random.random() < mutation_rate:
        index1 = random.randint(0, size - 1)
        index2 = random.randint(0, size - 1)
        permutation[index1], permutation[index2] = permutation[index2], permutation[index1]  # Swap
    else:
        available_jobs = set(map(frozenset, completePlay)) - set(map(frozenset, permutation))
        if available_jobs:
            index = random.randint(0, size - 1)
            permutation[index] = dict(random.choice(list(available_jobs)))  # Replace
    return permutation

def genetic_algorithm(completePlay, size, FunctionsBook, ran, per, mcost, weights):
    """Genetic Algorithm with debugging, dynamic mutation, and stagnation handling."""
    
    population_size = 100
    max_population_growth = 1.2
    min_population_reduction = 0.8
    max_stagnant_generations = 100  
    stagnant_generations = 0  
    mutation_rate = 0.1  

    minimal_permutation = per[:]  
    minimal_cost = ran  

    def initialize_population():
        population = [per[:]]  
        for _ in range(population_size - 1):
            population.append(random.sample(completePlay, size))  
        return population

    population = initialize_population()
    
    start_time = time.time()
    last_recorded_time = start_time
    end_time = start_time + 600  
    generation = 0  
    cost_history = [minimal_cost / ran]

    #print(f"🟢 Initial Cost (Greedy Solution): {minimal_cost / ran:.6f}")

    while time.time() < end_time:
        generation += 1  

        # Compute costs
        population_cost = [cost(perm, size, FunctionsBook, weights) for perm in population]

        # Sort population by cost (lower is better)
        population_sorted = sorted(zip(population, population_cost), key=lambda x: x[1])
        population, population_cost = zip(*population_sorted)
        population = list(population)
        population_cost = list(population_cost)

        # Track best solution
        if population_cost[0] < minimal_cost:
            minimal_cost = population_cost[0]
            minimal_permutation = population[0]
            stagnant_generations = 0  
            #print(f"✅ New Best Solution Found at Gen {generation}: {minimal_cost / ran:.6f}")
        else:
            stagnant_generations += 1  

        # Handle stagnation
        if stagnant_generations >= max_stagnant_generations:
            #print(f"🔄 Random Restart at Gen {generation} - No improvement for {max_stagnant_generations} generations.")
            population = initialize_population()  
            stagnant_generations = 0
            mutation_rate = 0.2  
            continue  

        # Record minimal cost every 60 seconds
        current_time = time.time()
        if current_time - last_recorded_time >= 60:
            cost_history.append(minimal_cost / ran)
            #print(f"📉 Time {int(current_time - start_time)}s: Best Cost {minimal_cost / ran:.6f}")
            last_recorded_time = current_time  

        # Adjust population size dynamically
        diversity = statistics.stdev(population_cost) if len(population_cost) > 1 else 0
        if diversity < 0.1 * minimal_cost:
            population_size = max(int(population_size * min_population_reduction), 50)
            mutation_rate = min(mutation_rate + 0.05, 0.3)  
        elif generation % 50 == 0:
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
                new_perm = mutate(population[parent], size, completePlay, mutation_rate)
            new_population.append(new_perm)

        population = new_population

    # Final measurement
    cost_history.append(minimal_cost / ran)
    return cost_history



def labor(size):
    constant_vector = np.random.randint(1, 11, size=size)
    for i in range(0, size, max(1, size // 10)):
        constant_vector[i] = np.random.randint(1, 11)
    return constant_vector

def machine_time(sequence_length, num_sequences):
    sequences = []
    mu = random.randint(sequence_length // 5, sequence_length // 3)
    tau = random.randint(sequence_length // 20, sequence_length // 10)
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

def bookFunction(size):
    FunctionsBook = {}
    FunctionsBook['Power_cons'] = [np.random.uniform(10, 60, size=size).tolist() for _ in range(size)]
    FunctionsBook['Ideal_cost'] = [np.full(size, 0).tolist()]
    FunctionsBook['Ideal_labor'] = [labor(size).tolist()]
    for i in range(1, 6):
        FunctionsBook[f'Machine_{i}'] = machine_time(size, size)
    FunctionsBook['Ranking'] = [list(range(1, size + 1))]
    return FunctionsBook

def generate_rankings(num_jobs, num_agents):
    total_sum_jobs = num_jobs * (num_jobs + 1) // 2
    job_rankings = np.zeros(num_jobs)
    for i in range(num_jobs):
        job_sum = min(total_sum_jobs, num_agents * (num_jobs - i))
        job_rankings[i] = job_sum
        total_sum_jobs -= job_sum
    np.random.shuffle(job_rankings)
    return job_rankings.tolist()

def generate_jobs(num_jobs, num_agents):
    job_rankings = generate_rankings(num_jobs, num_agents)
    bigJobs = []
    num_machines = 3  # Reduced number of machines
    for i in range(num_jobs):
        activated_machine = random.randint(1, num_machines)
        job_attributes = {
            'job_cost': np.random.uniform(50, 500),
            'power_consumption': np.random.uniform(5, 50),
            'labor_requirement': np.random.randint(1, 5),
            **{f"machine_{j}": 1 if j == activated_machine else 0 for j in range(1, num_machines + 1)},
            'job_ranking': job_rankings[i]
        }
        bigJobs.append(job_attributes)
    return bigJobs

def l1_distance(vec1, vec2):
    return sum(abs(a - b) for a, b in zip(vec1, vec2))

def swap_distance(vector1, vector2):
    swaps = 0
    for i in range(len(vector1)):
        if vector1[i] != vector2[i]:
            index = vector2.index(vector1[i])
            swaps += abs(vector1[i] - vector2[i])
            vector1[i], vector1[index] = vector1[index], vector1[i]
    return swaps

def get_values_by_key(tuples, key):
    return [t[key] for t in tuples if key in t]

def Scoring(solution, functions, distance, X):
    scores = [l1_distance(solution, func) if distance == 'l1' else swap_distance(solution, func) for func in functions]
    return min(scores)

def cost(permutation, X, FunctionsBook,weights ): 
    cost = 0

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

# Total number of jobs and jobs to rank
num_jobs = 5000
num_jobs_to_rank = 100
num_agents = 20
bigJobs = generate_jobs(num_jobs, num_agents)
FunctionsBook = bookFunction(num_jobs_to_rank)


# Graphing
gen = []
greed = []
simReg = []
instances = []
zRand = []

for i in range(10):  # Reduced number of instances
    instances.append(samples(500))  # Reduced sample size
for instance in instances:
    t = random.uniform(1, 3)
    z = random.uniform(0.5, 1.5)
    weights = [
        t, t, t, z,
        z, z, z, z, random.uniform(2, 3)
    ]

    ran, best_permutationz = zRandom(100, instance, num_jobs_to_rank, FunctionsBook,weights)
    permutation = greedy_algorithm(num_jobs_to_rank, instance, FunctionsBook, ran,weights)
    minimal_cost=cost(permutation, num_jobs_to_rank, FunctionsBook,weights)
    costArr2 = genetic_algorithm(instance, num_jobs_to_rank, FunctionsBook, ran,best_permutationz,1,weights)
    costArr3 = simulated_annealingReg(num_jobs_to_rank, instance, FunctionsBook, ran,best_permutationz,1,weights)
    gen.append(costArr2)
    greed.append(minimal_cost/ran)
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
