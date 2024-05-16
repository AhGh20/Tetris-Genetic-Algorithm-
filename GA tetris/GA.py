import pygame
import random, time
import numpy as np
import tetris_base as tetris

NUM_CHILD = 12

NUM_CHROMOSOMES = 12

NUM_GENERATIONS = 10
NUM_TRAINING_ITERATIONS = 400
NUM_TEST_ITERATIONS = 600


def create_chromosome(genes):
    return {'genes': genes, 'score': 1}



def create_population(num_pop, genes=7, lb=-1, ub=1):
    population = []
    for _ in range(num_pop):

        weights = np.random.uniform(lb, ub, size=(genes))

        chromosome = create_chromosome(weights)

        population.append(chromosome)

    return population


# print(create_population(12,7,-1,1))

# used the rullet wheel selection
def selection(population, num_selection=12):
    fitness = np.array([chromosome['score'] for chromosome in population])

    norm_fitness = fitness / fitness.sum()

    roulette_prob = np.cumsum(norm_fitness)

    selected_chromosomes = []
    while len(selected_chromosomes) < num_selection:
        pick = random.random()
        for index, chromosome in enumerate(population):
            if pick < roulette_prob[index]:
                selected_chromosomes.append(chromosome)
                break
    return selected_chromosomes

# def selection(population, selection_rate= 0.8):
#     fitness = np.array([chromosome['score'] for chromosome in population])
#
#     norm_fitness = fitness / fitness.sum()
#
#     roulette_prob = np.cumsum(norm_fitness)
#
#     selected_chromosomes = []
#
#     while len(selected_chromosomes) < int(selection_rate*len(population)):
#         pick = random.random()
#         for index, chromosome in enumerate(population):
#             if pick < roulette_prob[index]:
#                 selected_chromosomes.append(chromosome)
#                 break
#     return selected_chromosomes
#

def calc_best_move(chromosome, board, piece, show_game=False):
    best_X = 0  # Best X position
    best_R = 0  # Best rotation
    best_Y = 0  # Best Y position
    best_score = -100000  # Best score

    # Calculate the total the holes and blocks above holes before play
    num_holes_bef, num_blocking_blocks_bef = tetris.calc_initial_move_info(board)
    # Iterate through every rotation
    for r in range(len(tetris.PIECES[piece['shape']])):

        # Iterate through all positions
        for x in range(-2, tetris.BOARDWIDTH - 1):

            movement_info = tetris.calc_move_info(board, piece, x, r, num_holes_bef, num_blocking_blocks_bef)
            if movement_info[0]:
# score = w1 * height  + w2 * num_removed_lines +  w3 * number of hole + w4 * new_blocking_blocks + w5 *piece_sides + w6 * floor_sides + w7 *  wall_sides
                movement_score = sum(
                    chromosome['genes'][i] * movement_info[i + 1] for i in range(len(movement_info) - 1))
                if movement_score > best_score:
                    best_score = movement_score
                    best_X = x
                    best_R = r
                    best_Y = piece['y']

    if show_game:
        piece['y'] = best_Y
    else:
        piece['y'] = -2

    piece['x'] = best_X
    piece['rotation'] = best_R

    return best_X, best_R


#Random Resetting mutation
def mutation(chromosomes, mutation_rate):
    number_genes = sum(len(chromosome['genes']) for chromosome in chromosomes)
    mutate_number= int(number_genes * mutation_rate)
    mutation_positions = random.sample(range(1, number_genes + 1), mutate_number)

    for j in mutation_positions:
        index = (j-1) // 7
        gene_index = (j-1) % 7
        new_value = random.uniform(-1.0, 1.0)
        chromosomes[index]['genes'][gene_index] = new_value




#replace for the new genration
def replace_population(old_population, new_population):
#    replacement_point = (len(old_population) + len(new_population) ) *0.8
    replacement_point = 12

    population = old_population + new_population

    population = sorted(population, key=lambda x: x['score'], reverse=True)

    new_population = population[:replacement_point]

    #random.shuffle(new_population)

    return new_population



#used the one-point crossover
def crossover(selected_pop, desired_children=10, cross_rate=0.5):
    offspring = []
    child_count = 0

    while child_count < desired_children:
        # Select two random parents from the Selected population
        parent1, parent2 = random.sample(selected_pop, 2)
        child1 = {'genes': [], 'score': 0}
        child2 = {'genes': [], 'score': 0}
        crossover_point = random.randint(1, 6)
        child1['genes'] = np.concatenate((parent1['genes'][:crossover_point], parent2['genes'][crossover_point:]))
        child2['genes'] = np.concatenate((parent2['genes'][:crossover_point], parent1['genes'][crossover_point:]))

        offspring.extend([child1, child2])
        child_count += 2

    return offspring

def evaluate_population(population):
    for chromosome in population:
        game_state = run_game(chromosome, 100000, 1000000, False, 3)
        chromosome['score'] = game_state[2]


def run_genetic_algorithm(num_pop, num_weights=7, lb=-1, ub=1, selection_rate=12, cross_rate=0.4, mutation_rate=0.1,generations=10):
    population = create_population(num_pop, num_weights, lb, ub)
    evaluate_population(population)

    for i in range(generations):
        i += 1

        selected_chromosomes = selection(population, selection_rate)

        new_population = crossover(selected_chromosomes, 12,cross_rate)

        mutation(new_population, mutation_rate)

        evaluate_population(new_population)

        population = replace_population(population, new_population)
        print("best chromosome: {}  , generation: {}".format(population[0], i) )
    return population


# A=create_population(NUM_CHROMOSOMES)
# print("create_population :" ,A )
#
# B=selection(A , NUM_CHILD)
# print("SELECTED" , len(B))
#

def draw_game_on_screen(board, score, level, next_piece, falling_piece, chromosome):
    """Draw game on the screen"""

    tetris.DISPLAYSURF.fill(tetris.BGCOLOR)
    tetris.draw_board(board)
    tetris.draw_status(score, level)
    tetris.draw_next_piece(next_piece)

    if falling_piece != None:
        tetris.draw_piece(falling_piece)

    pygame.display.update()
    tetris.FPSCLOCK.tick(tetris.FPS)


def run_game(chromosome, speed, max_score=20000, no_show=False, test=2):
    tetris.FPS = int(speed)
    tetris.main()

    board = tetris.get_blank_board()
    last_fall_time = time.time()
    score = 0
    level, fall_freq = tetris.calc_level_and_fall_freq(score)
    falling_piece = tetris.get_new_piece()
    next_piece = tetris.get_new_piece()

    # Calculate best move
    calc_best_move(chromosome, board, falling_piece)

    num_used_pieces = 0
    removed_lines = [0, 0, 0, 0]  # Combos

    if test == 1:
        NUM_ITERATIONS = NUM_TRAINING_ITERATIONS
    elif test == 2:
        NUM_ITERATIONS = NUM_TEST_ITERATIONS
    else:
        NUM_ITERATIONS = 100000000
    win = False

    # Game loop
    while num_used_pieces < NUM_ITERATIONS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game exited by user")
                exit()

        if falling_piece == None:
            # No falling piece in play, so start a new piece at the top
            falling_piece = next_piece
            next_piece = tetris.get_new_piece()

            # Decide the best move based on your weights
            calc_best_move(chromosome, board, falling_piece, no_show)

            # Update number of used pieces and the score
            num_used_pieces += 1
            score += 1

            # Reset last_fall_time
            last_fall_time = time.time()

            if (not tetris.is_valid_position(board, falling_piece)):
                # GAME-OVER
                # Can't fit a new piece on the board, so game over.
                break

        if no_show or time.time() - last_fall_time > fall_freq:
            if (not tetris.is_valid_position(board, falling_piece, adj_Y=1)):
                # Falling piece has landed, set it on the board
                tetris.add_to_board(board, falling_piece)

                # Bonus score for complete lines at once
                # 40   pts for 1 line
                # 120  pts for 2 lines
                # 300  pts for 3 lines
                # 1200 pts for 4 lines
                num_removed_lines = tetris.remove_complete_lines(board)
                if (num_removed_lines == 1):
                    score += 40
                    removed_lines[0] += 1
                elif (num_removed_lines == 2):
                    score += 120
                    removed_lines[1] += 1
                elif (num_removed_lines == 3):
                    score += 300
                    removed_lines[2] += 1
                elif (num_removed_lines == 4):
                    score += 1200
                    removed_lines[3] += 1

                falling_piece = None
            else:
                # Piece did not land, just move the piece down
                falling_piece['y'] += 1
                last_fall_time = time.time()

        if (not no_show):
            draw_game_on_screen(board, score, level, next_piece, falling_piece,
                                chromosome)

        # Stop condition
        if (score > max_score):
            NUM_ITERATIONS = -1
            win = True


    # Save the game state
    game_state = [num_used_pieces, removed_lines, score, win]

    return game_state




