import main as ga
NUM_CHROMOSOMES = 12

pop = ga.run_genetic_algorithm(NUM_CHROMOSOMES ,generations=10)
pop.sort(key=lambda x: x['score'], reverse=True)
print(len(pop))

chromo = pop[:2]

print(chromo)
