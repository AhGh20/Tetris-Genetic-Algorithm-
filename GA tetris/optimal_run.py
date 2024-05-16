import main as ga

chromo={'genes': [-0.05651483,  0.93160698, -0.59719382, -0.16048557,  0.6871205 , 0.34392641,  0.72325149],'score': 1000039}

a=ga.run_game(chromo, speed=30000, max_score=50000, no_show=False, test=3)
print("___________________________________________")

if(a[3]):

    print("won :)")
else:
    ("lose :(")

print("___________________________________________")
print("number of pieces used :" + str(a[0]))
print("score : " , a[2])
# score = w1 * height  + w2 * num_removed_lines +  w3 * number of hole + w4 * new_blocking_blocks + w5 *piece_sides + w6 * floor_sides + w7 *  wall_sides

#the chromo play 600 games and give us the best score
#max_score = -1
# for _ in range(600):
#     game_state = ga.run_game(chromo, speed=100000, max_score=10000, no_show=False, test=3)
#     if game_state[2] > max_score:
#         max_score = game_state[2]
#
#
# print("max score is :", max_score)



