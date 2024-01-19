import itertools
from collections import Counter
from operator import itemgetter
import numpy as np
import random
import math
import time
import statistics
import matplotlib.pyplot as plt
import simplejson as json
# Load tracks from JSON file
with open('tracks.json') as f: # using the Spotify API to retrive track data from: https://open.spotify.com/playlist/1G8IpkZKobrIlXcVPoSIuf
    bigPlay = json.load(f)

# 0.1.1 Add Optimal Functions and Scoring Method
def pol(degree, i, begin, end, minSlope, maxSlope, minimum, maximum):
    listPol = []

    if degree != 0:
        for _ in range(20):
            Pol = []
            indexArr = []
            plusMinus = random.sample([-1.0, 1.0], 1)[0]
            val = random.uniform(begin, end)
            Pol.append(val)

            for j in range(degree - 1):
                index = random.randint(2, math.floor(i / 2))
                
                while (len(indexArr) > 0) and (index - indexArr[-1] < 2):
                    index = random.randint(indexArr[-1], i - 2)
                indexArr.append(index)

                for a in range(i - 1):
                    flag = 0
                    increment = random.uniform(minSlope, maxSlope)

                    if val + plusMinus * increment <= minimum:
                        val = minimum
                        flag = 1
                    elif val + plusMinus * increment >= maximum:
                        val = maximum
                        flag = 1

                    if a in indexArr and flag == 0:
                        plusMinus = plusMinus * (-1)
                    val += increment * plusMinus

                Pol.append(val)

            listPol.append(Pol)
    else:
        for _ in range(20):
            val = random.uniform(begin, end)
            Pol = [val for _ in range(i)]
            listPol.append(Pol)

    return listPol

def bookFunction(i):
    FunctionsBook = {}
    FunctionsBook['bpmStatic'] = pol(0, i, 120, 180, 0, 0, 120, 180)
    FunctionsBook['bpmMove1'] = pol(1, i, 120, 180, 1, 10, 40, 400)
    FunctionsBook['bpmMove2'] = pol(2, i, 120, 180, 1, 10, 40, 400)
    FunctionsBook['bpmMove3'] = pol(3, i, 120, 180, 1, 10, 40, 400)
    FunctionsBook['energyStatic'] = pol(0, i, 0.2, 1, 10, 0, 0.2, 0.95)
    FunctionsBook['energyMove1'] = pol(1, i, 0.2, 0.7, 0.02, 0.05, 0.2, 0.95)
    FunctionsBook['energyMove2'] = pol(2, i, 0.2, 0.7, 0.02, 0.05, 0.2, 0.95)
    FunctionsBook['energyMove3'] = pol(3, i, 0.2, 0.7, 0.02, 0.05, 0.2, 0.95)
    FunctionsBook['danceability'] = pol(0, i, 0.3, 0.8, 0.02, 0.05, 0.2, 0.95)
    FunctionsBook['votes'] = pol(0, i, 20, 20, 0, 0, 20, 20)
    
    return FunctionsBook

def l1_distance(vec1, vec2):
    return sum(abs(vec1[i] - vec2[i]) for i in range(len(vec1)))

def l2_distance(vec1, vec2):
    return math.sqrt(sum((vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))))

def Scoring(solution, functions, distance, X):
    scores = []

    for i in range(len(functions)):
        if distance == 'l2':
            scores.append(l2_distance(solution, functions[i]))
        else:
            scores.append(l1_distance(solution, functions[i]))

    return min(scores)

def get_values_by_key(tuples, key):
    values = []

    for t in tuples:
        if key in t:
            values.append(t[key])

    return values

def cost(permutation, X, FunctionsBook):
    cost = 0
    zeroDeg = random.uniform(1, 3)
    firstDeg = random.uniform(1, zeroDeg)
    secDeg = random.uniform(1, firstDeg)
    thirdDeg = random.uniform(1, secDeg)

    weights = [zeroDeg, firstDeg, secDeg, thirdDeg, zeroDeg, firstDeg, secDeg, thirdDeg,
               random.uniform(3, 4), random.uniform(4, 5)]

    permutationBook = {}
    j = 0

    for i in FunctionsBook.keys():
        permutationBook[i] = get_values_by_key(permutation, i.lower())

    for i in permutationBook.keys():
        cost += weights[j] * Scoring(permutationBook[i], FunctionsBook[i], 'l1', X)
        j += 1

    return cost

def scoringSim(X):
    random_numbers = []
    remaining_sum = 20 * X

    for i in range(700):
        random_number = random.randint(0, min(20, remaining_sum // (701 - i)))
        random_numbers.append(random_number)
        remaining_sum -= random_number

    while remaining_sum > 0:
        index = random.randint(0, 699)

        if random_numbers[index] < 20:
            random_numbers[index] += 1

        remaining_sum -= 1
        random_numbers.append(remaining_sum)

    return random_numbers[0:700]

def samples(i, size):
    votes = scoringSim(size)
    instance = random.sample(bigPlay, i)

    for p in range(i):
        instance[p]['votes'] = votes[p]
        instance[p]['number'] = p

    return instance

# 0.1.2 Z Random
def zRandom(z, sol, i, FunctionsBook):
    scores = []

    for j in range(z):
        x = cost(random.sample(sol, i), i, FunctionsBook)
        scores.append(x)

    return min(scores)

# 0.1.3 Greedy
def greedy_algorithm(size, completePlay, FunctionsBook, ran):
    start_time = time.time()
    permutation = []
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

        permutation.append(minimal_element)
        minimal_cost = cost(permutation, size, FunctionsBook) / ran

    return minimal_cost

# 0.1.4 Simulated Annealing
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
    target_acceptance_rate = 0.5

    while time.time() < end_time:
        new_permutation = permutation
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
            cooling_rate *= 1.05
        else:
            cooling_rate *= 0.95

        temperature *= 1 - cooling_rate
        counter += 1

        if counter == 1000:
            permutation = random.sample(completePlay, size)
            old_cost = cost(permutation, size, FunctionsBook)
            counter = 0

        if int((time.time() - elapsed_time) / 60) == len(cost_history):
            cost_history.append(minimal_cost / ran)

    return cost_history

# 0.1.5 Genetic
def adaptive_crossover_prob(generation):
    initial_crossover_prob = 0.85
    min_crossover_prob = 0.5
    max_generation = 1000
    crossover_prob = max(initial_crossover_prob - (generation / max_generation) * 
                        (initial_crossover_prob - min_crossover_prob), min_crossover_prob)

    return crossover_prob

def crossover(permutation1, permutation2, size):
    crossover_point = random.randint(1, size - 1)
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
    min_population_size = 100
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
        fitness = []

        for cost_val in population_cost:
            fitness.append(1 / (cost_val - minimal_cost + 1))

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

# 0.1.6 Graphing
gen = []
greed = []
simReg = []
instances = []
zRand = []
FunctionsBook = bookFunction(250)

# Generate samples
for _ in range(10):
    instances.append(samples(700, 250))

# Run algorithms and collect results
for instance in instances:
    ran = zRandom(100, instance, 250, FunctionsBook)
    minimal_cost = greedy_algorithm(250, instance, FunctionsBook, ran)
    costArr2 = genetic_algorithm(instance, 250, FunctionsBook, ran)
    costArr3 = simulated_annealingReg(250, instance, FunctionsBook, ran)
    
    gen.append(costArr2)
    greed.append(minimal_cost)
    simReg.append(costArr3)
    zRand.append(ran)

# Calculate means
gen_mean = [sum(x)/10 for x in zip(*gen)]
simReg_mean = [sum(x)/10 for x in
