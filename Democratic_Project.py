import numpy as np
import random
import math
import time
import statistics
import matplotlib.pyplot as plt
import itertools

# Function definitions

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
    # Set the initial permutation
    permutation = []
    # Set the minimal cost to a high value
    minimal_cost = float('inf')
    for i in range(size):
        minimal_cost_element = float('inf')
        minimal_element = None
        
        for element in completePlay:
            if element not in permutation:
                permutation_cost = cost(permutation + [element], size, FunctionsBook)
                if permutation_cost < minimal_cost_element:
                    minimal_cost_element = permutation_cost
                    minimal_element = element
        
        if minimal_element is not None:
            # Add the minimal element to the permutation
            permutation.append(minimal_element)
            
            # Update the minimal cost
            minimal_cost = cost(permutation, size, FunctionsBook) / ran
    
    return minimal_cost



def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)

def simulated_annealingReg(size, completePlay, FunctionsBook, ran):
    cost_history = []
    counter = 0
    temperature = 1000
    minimal_permutation = None
    cooling_rate = 0.003
    permutation = random.sample(completePlay, size)
    old_cost = cost(permutation, size, FunctionsBook)
    minimal_cost = float('inf')
    end_time = time.time() + 600
    elapsed_time = time.time()
    improvement_counter = 0
    total_counter = 0
    target_acceptance_rate = 0.5  # Set the target acceptance rate to 50%
    
    while time.time() < end_time:
        new_permutation = permutation[:]
        
        index1 = random.randint(0, size - 1)
        index2 = 0
        t = new_permutation[index1]
        
        track = random.sample(completePlay, 1)
        while track == t:
            track = random.sample(completePlay, 1)
        
        if track in permutation:
            index2 = new_permutation.index(track)
            new_permutation.remove(track)
            new_permutation.remove(t)
            new_permutation.insert(index1, track)
            new_permutation.insert(index2, t)
        else:
            new_permutation.remove(t)
            new_permutation.insert(index1, track)
        
        new_cost = cost(new_permutation, len(new_permutation), FunctionsBook)
        acceptance_prob = acceptance_probability(old_cost, new_cost, temperature)
        random_number = random.random()
        
        if random_number < acceptance_prob:
            permutation = new_permutation
            old_cost = new_cost
            improvement_counter += 1
        
        if new_cost < minimal_cost:
            minimal_cost = new_cost
            minimal_permutation = new_permutation
        
        total_counter += 1
        acceptance_rate = improvement_counter / total_counter
        
        if acceptance_rate > target_acceptance_rate:
            cooling_rate *= 1.05  # Increase cooling rate to converge faster
        else:
            cooling_rate *= 0.95  # Decrease cooling rate to explore more of the search space
        
        temperature *= 1 - cooling_rate
        counter += 1
        
        if counter == 1000:
            permutation = random.sample(completePlay, size)
            old_cost = cost(permutation, size, FunctionsBook)
            counter = 0
        
        if int((time.time() - elapsed_time) / 60) == len(cost_history):
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

def genetic_algorithm(completePlay, size, FunctionsBook, ran):
    cost_history = []
    # Set the minimum population size
    min_population_size = 100
    # Set the maximum population size
    max_population_size = 1000
    minimal_cost = float('inf')
    minimal_permutation = None
    population_size = min_population_size
    
    population = []
    for _ in range(population_size):
        permutation = random.sample(completePlay, size)
        population.append(permutation)
    
    end_time = time.time() + 600
    elapsed_time = time.time()
    generation = 0
    
    while time.time() < end_time:
        population_cost = []
        generation += 1
        
        for permutation in population:
            c = cost(permutation, size, FunctionsBook)
            population_cost.append(c)
        
        population_sorted = sorted(zip(population, population_cost), key=lambda x: x[1])
        population, population_cost = zip(*population_sorted)
        population = list(population)
        population_cost = list(population_cost)
        
        if population_cost[0] < minimal_cost:
            minimal_cost = population_cost[0]
            minimal_permutation = population[0]
        
        if int((time.time() - elapsed_time) / 60) == len(cost_history):
            cost_history.append(minimal_cost / ran)
        
        avg_cost = sum(population_cost) / len(population_cost)
        std_cost = statistics.stdev(population_cost)
        
        fitness = [1 / (cost_val - minimal_cost + 1) for cost_val in population_cost]
        fitness_sum = sum(fitness)
        fitness = [f / fitness_sum for f in fitness]
        
        neff = 1 / sum([f ** 2 for f in fitness])
        
        if neff > population_size:
            population_size = min(max_population_size, int(population_size * 1.2))
        
        while len(population) < population_size:
            crossover_prob = adaptive_crossover_prob(generation)
            random_number = random.random()
            
            if random_number < crossover_prob:
                parent1 = random.randint(0, len(population) - 1)
                parent2 = random.randint(0, len(population) - 1)
                permutation = crossover(population[parent1], population[parent2], size)
            else:
                parent = random.randint(0, len(population) - 1)
                permutation = mutate(population[parent], size, completePlay)
            
            population.append(permutation)
    
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
    minimal_cost = greedy_algorithm(250, instance, FunctionsBook, ran)
    costArr2 = genetic_algorithm(instance, 250, FunctionsBook, ran)
    costArr3 = simulated_annealingReg(250, instance, FunctionsBook, ran)
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
    total1 = 0
    total2 = 0
    total3 = 0
    for b in range(10):
        total2 += simReg[b][a]
        total3 += gen[b][a]
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

def simulated_annealingTime(size, completePlay, FunctionsBook, ran):
    cost_history = []
    counter = 0
    temperature = 1000
    minimal_permutation = None
    cooling_rate = 0.003
    permutation = random.sample(completePlay, size)
    old_cost = cost(permutation, size, FunctionsBook)
    minimal_cost = float('inf')
    end_time = time.time() + 600
    elapsed_time = time.time()
    improvement_counter = 0
    total_counter = 0
    target_acceptance_rate = 0.5  # Set the target acceptance rate to 50%
    
    while time.time() < end_time:
        new_permutation = permutation[:]
        
        index1 = random.randint(0, size - 1)
        index2 = 0
        t = new_permutation[index1]
        
        track = random.sample(completePlay, 1)
        while track == t:
            track = random.sample(completePlay, 1)
        
        if track in permutation:
            index2 = new_permutation.index(track)
            new_permutation.remove(track)
            new_permutation.remove(t)
            new_permutation.insert(index1, track)
            new_permutation.insert(index2, t)
        else:
            new_permutation.remove(t)
            new_permutation.insert(index1, track)
        
        new_cost = cost(new_permutation, len(new_permutation), FunctionsBook)
        acceptance_prob = acceptance_probability(old_cost, new_cost, temperature)
        random_number = random.random()
        
        if random_number < acceptance_prob:
            permutation = new_permutation
            old_cost = new_cost
            improvement_counter += 1
        
        if new_cost < minimal_cost:
            minimal_cost = new_cost
            minimal_permutation = new_permutation
        
        if new_cost<ran*0.5:
            return time.time()-elapsed_time
        
        total_counter += 1
        acceptance_rate = improvement_counter / total_counter
        
        if acceptance_rate > target_acceptance_rate:
            cooling_rate *= 1.05  # Increase cooling rate to converge faster
        else:
            cooling_rate *= 0.95  # Decrease cooling rate to explore more of the search space
        
        temperature *= 1 - cooling_rate
        counter += 1
        
        if counter == 1000:
            permutation = random.sample(completePlay, size)
            old_cost = cost(permutation, size, FunctionsBook)
            counter = 0
        
        if int((time.time() - elapsed_time) / 60) == len(cost_history):
            cost_history.append(minimal_cost / ran)
    
    return time.time()-elapsed_time

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
    
instances = []
dicT={}
for i in range(10):
    instances.append(samples(700))

for c in range(25,251,25):
    dicT[c]=[]
    FunctionsBook = bookFunction(c)
    for instance in instances:
        ran = zRandom(100, instance,c, FunctionsBook)
        dicT[c].append(simulated_annealingTime(c, instance, FunctionsBook, ran))
