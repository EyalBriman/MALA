
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
