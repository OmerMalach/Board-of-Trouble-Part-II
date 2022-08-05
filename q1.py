import copy
import networkx as nx
import random
import math
import numpy as np
import tqdm


class Grid:
    def __init__(self, grid, agents, targets):
        self.grid = grid
        self.agents = agents  # containing all nodes preforming as agents
        self.targets = targets  # containing all nodes preforming as targets
        self.board_number = 1  # for printing use

    def print_board(self, h, goal_board_number, search_method, consider_list, fake, fake_counter, prob, mutated):
        if self.board_number == 1 and not search_method == 5:
            print('Board ' + str(self.board_number) + ' (starting position):')  # printing starting title
            self.board_number += 1
        elif fake and self.board_number == 2 and h > 0:  # for bag printing (k-beam)
            alphabet_list = ['a', 'b', 'c']
            print('Board ' + str(self.board_number) + alphabet_list[fake_counter] + ':')
        elif search_method == 5 and fake:  # for parent printing (genetic)
            if prob != 0:
                print('Starting board ' + str(fake_counter) + ' probability of selection from population: <' + str(prob)
                      + '>')
            else:
                print('Result board (mutation happened: <' + mutated + '>):')  # title for child print
        elif self.board_number == goal_board_number and goal_board_number > 1 and not fake:
            print('Board ' + str(self.board_number) + ' (goal position):')  # printing goal title
        else:
            print('Board ' + str(self.board_number) + ':')  # printing title
            self.board_number += 1
        title = '   '
        for k in range(len(self.grid)):  # creating the numbers above the board
            title += (' ' + str(k + 1))
        for i in range(len(self.grid)):  # responsible on printing the actual board
            line = ''
            if i == 0:
                print(title)
            for j in range(len(self.grid)):  # responsible on printing the actual board
                t = ''
                if self.get_grid()[i][j].get_state() == 1:
                    t = '@ '
                if self.get_grid()[i][j].get_state() == 2:
                    t = '* '
                if self.get_grid()[i][j].get_state() == 0:
                    t = '  '
                if self.get_grid()[i][j].get_state() == 3:  # indicating dna from DAD
                    t = 'D '
                if self.get_grid()[i][j].get_state() == 4:  # indicating dna from MOM
                    t = 'M '
                line += t
            print(i + 1, ':', line)
        if self.board_number == 3 and h > 0 and search_method == 1:  # in case bool is set to true> show heuristics
            print('Heuristic : ' + str(h))
        if self.board_number == 2 and search_method == 3 and h > 0 and consider_list is not None and goal_board_number \
                != self.board_number == 2:
            for consideration in consider_list:
                if type(consideration[1]) is not str:
                    print('action:', '(', str(consideration[0].x + 1), ',', str(consideration[0].y + 1), ')-> (',
                          str(consideration[1].x + 1), ',', str(consideration[1].y + 1), '); probability: ',
                          str(consideration[2]))  # printing considerations (simulated)
                else:
                    print('action:', '(', str(consideration[0].x + 1), ',', str(consideration[0].y + 1), ')-> (',
                          consideration[1], '); probability: ', str(consideration[2]))
        print('_____')

    def get_grid(self):
        return self.grid

    def get_agents(self):
        return self.agents

    def get_targets(self):
        return self.targets


class Node:
    def __init__(self, x, y, my_state, total_row):

        self.x = x
        self.y = y
        self.my_state = my_state
        self.total_row = total_row
        self.f = 0
        self.g = 0
        self.h = 1000
        self.neighbors = []
        self.consider = []
        self.previous = None
        self.visited = False
        self.neighbors_direction = []
        self.route_evolution = []

    def __repr__(self):
        return f'({self.x},{self.y})->'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(('row', self.x,
                     'col', self.y))

    def update_neighbors(self, grid, targets):  # method is responsible of linking node to each other
        if self.x == len(grid) and self.y == len(grid):  # meaning were dealing with exit node
            # (exit node are generated in case there are more agents than targets.. meaning some of the agents
            pass

        else:
            if (self.x < len(grid) - 1) and ((grid[self.x + 1][self.y].get_state() == 0) or
                                             (grid[self.x + 1][self.y].get_state() == 2)):  # down
                self.neighbors.append(grid[self.x + 1][self.y])
                if self.x < len(grid) - 1:
                    self.neighbors_direction.append((grid[self.x + 1][self.y], np.array([1, 0])))

            if (self.x > 0) and ((grid[self.x - 1][self.y].get_state() == 0) or
                                 (grid[self.x - 1][self.y].get_state() == 2)):  # up
                self.neighbors.append(grid[self.x - 1][self.y])
                if self.x > 0:
                    self.neighbors_direction.append((grid[self.x - 1][self.y], np.array([-1, 0])))

            if (self.y < len(grid) - 1) and ((grid[self.x][self.y + 1].get_state() == 0) or
                                             (grid[self.x][self.y + 1].get_state() == 2)):  # right
                self.neighbors.append(grid[self.x][self.y + 1])
                if self.y < len(grid) - 1:
                    self.neighbors_direction.append((grid[self.x][self.y + 1], np.array([0, 1])))

            if (self.y > 0) and ((grid[self.x][self.y - 1].get_state() == 0) or
                                 (grid[self.x][self.y - 1].get_state() == 2)):  # left
                self.neighbors.append(grid[self.x][self.y - 1])
                if self.y > 0:
                    self.neighbors_direction.append((grid[self.x][self.y - 1], np.array([0, -1])))
            if self.x == len(grid) - 1:  # if the node were looking at is in the last row
                for j in range(len(targets)):  # for each target
                    if targets[j].get_x() == len(grid):  # if the target is a exit node (exit node.get_x is set to the
                        # length of the board)
                        self.neighbors.append(targets[j])  # connect this node to the exit node
                        self.neighbors_direction.append((targets[j], np.array([1, 0])))
                        break

    def get_x(self):
        return self.x

    def get_h(self):
        return self.h

    def get_y(self):
        return self.y

    def get_state(self):
        return self.my_state

    def set_state(self, new_state):
        self.my_state = new_state

    def get_neighbors(self):
        return self.neighbors

    def get_total_row(self):
        return self.total_row

    def reset_previous(self):
        self.previous = None

    def set_previous(self, previous):
        self.previous = previous


def get_neighbors(grid, targets):  # method is responsible of iterating the grid and updating
    # each node possible neighbors
    for target in targets:
        if target.get_x() == len(grid):  # in case one of our targets is a exit node, its row and col would be set to
            # the length of the board (to identify them)
            target.update_neighbors(grid, targets)  # here were calling update neighbors inorder to connect the
            # exit node to the last row thus making them neighbors

    for i in range(len(grid)):
        for j in range(len(grid)):
            grid[i][j].update_neighbors(grid, targets)

    pass


class Route:  # object made for genetic solve
    def __init__(self, node, end, mutation_rate, pop_size):
        self.head = np.array([node.x, node.y])
        self.tail = np.array([node.x, node.y])
        self.start = node
        self.end = end
        self.current = node
        self.dna = []
        self.heritage = []
        self.mutation_rate = mutation_rate
        self.fit = 0
        self.generation = 0
        self.pop_size = pop_size
        self.crossover_index = 0
        self.mutated = False
        self.selection_probability = 0
        self.selections = []

    def update_head(self):  # method recalculates head position
        self.head[0] = self.start.x  # reset head coordinates
        self.head[1] = self.start.y
        for i in self.dna:
            self.head += i
        return self

    def path_legit(self):  # check for proper path
        self.head[0] = self.start.x  # reset head coordinates
        self.head[1] = self.start.y
        node = self.start
        flag = False
        for i in self.dna:
            self.head += i
            if (self.head[0] == self.start.total_row or self.head[1] == self.start.total_row) \
                    and self.end.total_row == self.end.x:  # preventing from algo to cheat and skip through the
                return True  # exit node
            else:
                for j in node.neighbors:
                    if self.head[0] == j.x and self.head[1] == j.y:
                        node = j
                        flag = True
                        break
                    else:
                        flag = False
                        continue
        return flag


def agent_counter(board):  # method counts the number of agents/targets for board checking
    agents_counter = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if isinstance(board[i][j], Node):
                if board[i][j].get_state() == 2:
                    agents_counter += 1
            else:
                if board[i][j] == 2:
                    agents_counter += 1

    return agents_counter


def board_is_legit(node_grid, goal_board):  # method is responsible of making sure that the forcefields are in the
    # same place and that the number of targets is less than the number of agents
    grid = node_grid
    start_agents_counter = agent_counter(node_grid)
    goal_agents_counter = agent_counter(goal_board)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (grid[i][j].get_state() == 1 and goal_board[i][j] != 1) or (
                    grid[i][j].get_state() != 1 and goal_board[i][j] == 1):
                # making sure that the forcefields are in the same place -
                # if the locations are not identical >> meaning the goal board isn't valid
                print("the forcefields spots between the two boards are not identical")
                return False

    return start_agents_counter >= goal_agents_counter


def make_grid(start_board, goal_board):  # method is responsible of generating Grid object
    grid = []
    agents = []
    targets = []
    for i in range(len(start_board)):
        grid.append([])
        for j in range(len(start_board)):
            spot_initial_value = start_board[i][j]  # getting values from starting board
            node = Node(i, j, spot_initial_value, len(start_board))  # creating nodes
            grid[i].append(node)  # filling the grid (2D array) in nodes
            if spot_initial_value == 2:
                agents.append(node)  # if the initial value at the stating board is 2 add the node into agents list
            if goal_board[i][j] == 2:
                targets.append(node)  # if the initial value at the goal board is 2 add the node into targets list
    exit_node = Node(len(start_board), len(start_board), 0, len(start_board))  # creating exit node in case of more
    # agents than targets
    for k in range(len(agents) - len(targets)):
        targets.append(exit_node)  # the diff between the two lists is the number of extra targets we need to add
        # into the targets list
    output = Grid(grid, agents, targets)  # finally > creating a grid object
    return output


def find_path(starting_board, goal_board, search_method, detail_output):
    node_grid = make_grid(starting_board, goal_board)  # creating grid object
    get_neighbors(node_grid.get_grid(), node_grid.get_targets())  # updates all nodes neighbors
    if board_is_legit(node_grid.get_grid(), goal_board):  # call for a simple board check
        multi_agent_handler(node_grid, detail_output, search_method)  # call for a solve
    else:
        print("one of the boards is inconsistent with the rules of the game. ")
        print("please try again with a valid input of boards.")


class AStar:

    @staticmethod
    def clean_open_set(open_set, current_node):  # method is responsible of popping nodes out of the open-set as part of
        # A-star algorithm
        for i in range(len(open_set)):
            if open_set[i] == current_node:
                open_set.pop(i)
                break

        return open_set

    @staticmethod
    def h_score(current_node, end):  # calculating manhattan distance between two nodes
        if end.get_x() == current_node.get_total_row() and end.get_y() == current_node.get_total_row():  # if the end
            #  node were currently looking at is in fact a exit node don't mind the y coordinate when calculating the
            #  distance
            distance = abs(current_node.x - end.x)
        else:
            distance = abs(current_node.x - end.x) + abs(current_node.y - end.y)

        return distance

    @staticmethod
    def start_path(current_node, end):  # the actual a* algorithm - takes in 2 nodes.
        final_path = []
        current_node.reset_previous()
        if current_node.get_x() == end.get_x() and current_node.get_y() == end.get_y():  # if we already have a agent on
            # a target
            final_path.append(current_node)  # add the node into final path ( a list of nodes that symbolizes the path)
        else:
            open_set = [current_node]  # add the start node into the open set
            closed_set = []
            current_node.g = 0  # reset the start node g cost to 0
            while len(open_set) != 0:  # as long as the open set is not empty
                best_way = 0
                for i in range(len(open_set)):  # the power of the a-star algorithm is this line below
                    if open_set[i].f < open_set[best_way].f:  # best way gets the node in open set with the lowest
                        # f value (f= h+g) instead of using priority queue
                        best_way = i
                current_node = open_set[best_way]  # current node gets to be the node with the lowest f cost
                if current_node.get_x() == end.get_x() and current_node.get_y() == end.get_y():  # if the current node
                    # were looking at is actually the end node
                    temp = current_node
                    final_path.append(temp)  # add temp node into the list called final path
                    while temp.previous:  # while there is a node that led to the temp node
                        final_path.insert(0, temp.previous)  # insert the previous node into the list
                        temp = temp.previous
                    pass  # when the thread gets to this point is "breaks" the  "while len(open_set) != 0:" and
                    # final path gets returned
                open_set = AStar.clean_open_set(open_set, current_node)  # pop current node from open set
                closed_set.append(current_node)  # and append it to close set
                neighbors = current_node.neighbors  # gets the list of neighbor nodes
                for neighbor in neighbors:  # for each of the neighbors in that list
                    if neighbor in closed_set:  # if its already been evaluated continue
                        continue
                    else:
                        temp_g = current_node.g + 1  # the cost of another step
                        control_flag = 0  # reset control flag to 0
                        for k in range(len(open_set)):
                            if neighbor.x == open_set[k].x and neighbor.y == open_set[k].y:  # if the neighbor that were
                                # looking at is already in the open set
                                if temp_g < open_set[k].g:  # if the current temp_g is lower than the g of
                                    # that neighbor (meaning we found a chipper way) then update that node g h and
                                    # f values
                                    open_set[k].g = temp_g
                                    open_set[k].h = AStar.h_score(open_set[k], end)
                                    open_set[k].f = open_set[k].g + open_set[k].h
                                    open_set[k].previous = current_node  # update the previous node to the current_node
                                    # in order to reverse engineer the optimal path
                                else:  # if the neighbor that were looking at is already in the open set but the temp
                                    # value is not cheaper there is no need to update it >> control flag = 1 and we wont
                                    # enter the else that adds the node into the open set (its already there and the
                                    # values don't need to get updated)
                                    pass
                                control_flag = 1
                        if control_flag == 1:  # normally the flag is set to zero
                            pass
                        else:  # and we update the neighbor g h f and previous values
                            neighbor.g = temp_g
                            neighbor.h = AStar.h_score(neighbor, end)
                            neighbor.f = neighbor.g + neighbor.h
                            neighbor.previous = current_node
                            open_set.append(neighbor)  # and add the node to the open set
        return final_path  # finally return the optimal path between the start node and the end node


def generate_random_move(current_node, end):  # method responsible of returning a random step (used after restarting
    # hill climbing)
    copied_list_of_neighbor = copy.deepcopy(current_node.get_neighbors())
    for node in copied_list_of_neighbor:
        if node.x == node.get_total_row() and end.x != node.get_total_row():
            copied_list_of_neighbor.remove(node)
    if len(copied_list_of_neighbor) > 0:
        move = random.choice(copied_list_of_neighbor)
        real_move = matcher(move, current_node)
    else:
        real_move = None
    return real_move


def matcher(move, current_node):  # method is used to match the actual nodes after deep copy
    for node in current_node.get_neighbors():
        if move.x == node.x and move.y == node.y:
            return node


def find_minimum_step(current_node, end, final_path, not_fake):  # used in hill Climbing
    # method responsible of returning the minimal step possible
    possible_moves = []
    min_move = None
    for neighbor in current_node.get_neighbors():  # iterate over all neighbors
        if not_fake and neighbor in final_path:
            # if the neighbor (node/step) we are looking at is already a part of the path
            continue  # move on to the next neighbor
        else:
            neighbor.h = AStar.h_score(neighbor, end)
            possible_moves.append(neighbor)
    if len(possible_moves) > 0:
        min_move = min(possible_moves, key=lambda step: step.h)
    return min_move


def hill_climb(current_node, end):  # hill climb search algorithm
    restart_counter = 0
    final_path = []
    best_move = None
    while restart_counter <= len(current_node.get_neighbors()):  # restart takes place if we didn't reach the goal node
        final_path = [current_node]
        best_score = AStar.h_score(current_node, end)  # calculating starting node h score
        if best_score == 0:  # if the starting node is already located in the goal position
            break  # if we reached this point break the outer while loop
        else:  # meaning the start node is not positioned at the goal position
            if restart_counter == 0:  # in the first round we will try and start the path with the minimal
                # step possible
                best_move = find_minimum_step(current_node, end, final_path, True)
                # method responsible of returning minimal
                # neighbor (minimal step)
                final_path.append(best_move)  # add the second node into the path
            if restart_counter > 0:  # if we didn't reach the goal node in first round(when restart_counter=0)
                # it means
                # that the path we took in round 0 wont reach the goal node
                best_move = generate_random_move(current_node, end)  # we generate a random first step
                final_path.append(best_move)  # add the random first step into the path
            while True:
                best_score = AStar.h_score(best_move, end)  # calculate the h score of the first step
                if best_score == 0:  # meaning the step we just made was towards the goal node
                    break  # break the inner while loop when we reached our goal
                next_move = find_minimum_step(best_move, end, final_path, True)  # finding the minimal step
                if next_move is not None:
                    score = AStar.h_score(next_move, end)  # calculate the the next step h score
                    if score < best_score:  # if it is a move in the right direction
                        final_path.append(next_move)  # add the next node into the path
                        best_move = next_move  # make the next move (next node) into our current node (best move)
                    else:  # if we didn't improve > start the next round
                        restart_counter += 1
                        final_path.clear()
                        break  # breaking the inner loop
                else:  # if we didn't improve > start the next round
                    restart_counter += 1
                    final_path.clear()
                    break  # breaking the inner loop
        if best_score == 0:  # if condition is true> we reached our goal> there is no need to search more and we should
            break
    return final_path


def simulated_annealing(start, end):
    temperature = 100
    alpha = 0.9729
    final_path = [start]
    start.h = AStar.h_score(start, end)  # calculating starting node h score
    current = start
    counter = 0
    rand = 0
    probability_to_fail = 0
    p = 0
    consider_moves = []
    while temperature > 1:
        if current.h == 0:  # if the starting node is already located in the goal position
            return final_path
        best_next = generate_random_move(current, end)  # we generate a random first step
        if best_next is not None:
            best_next.h = AStar.h_score(best_next, end)  # calculate the the next step h score
            if temperature > 0.001:
                p = math.exp(current.h - best_next.h / temperature)
                rand = round(random.uniform(0, 1), 3)
                probability_to_fail = round(1 - rand, 3)
            if best_next.h < current.h:  # if it is a move in the right direction
                if counter == 0:
                    consider_moves.append((current, best_next, 1))  # for printing consideration later on
                    current.consider.append(consider_moves)
                    counter += 1
                final_path.append(best_next)  # add the next node into the path
                current = best_next
            elif rand < p:  # taking "bad" choices in the beginning of the run
                if counter == 0:
                    consider_moves.append((current, best_next, probability_to_fail))  # for printing consideration later
                    current.consider.append(consider_moves)
                    counter += 1
                final_path.append(best_next)  # add the next node into the path
                current = best_next
            temperature *= alpha
            if counter == 0:  # as long as the counter == 0 > meaning no step has been taken> keep adding into
                # consider_move
                consider_moves.append((current, best_next, rand))
        else:
            break
    if current.h != 0:
        final_path.clear()
    return final_path


def reset_visiting(node_grid):  # method is used after every full iteration (finding a path btw agent to target)
    size = len(node_grid.get_grid())  # of k-beam
    for i in range(size):
        for j in range(size):
            node_grid.get_grid()[i][j].visited = False
            node_grid.get_grid()[i][j].h = 1000
            node_grid.get_grid()[i][j].previous = None
    for target in node_grid.get_targets():
        if target.x == size and target.y == size:
            target.visited = False
            target.previous = None


def update_h_score(open_list, current, end):  # used in k-beam to update the h score off all node in open_list
    for step in open_list:
        step.h = AStar.h_score(step, end)
        if step in current.neighbors:
            if circle_detector(step, current):
                step.previous = current


def circle_detector(step, current):  # method prevents a circle from happening
    temp = current
    old = step.previous
    step.previous = current
    while temp.previous is not None:
        if temp is not step:
            temp = temp.previous
        else:
            step.previous = old
            return False
    return True


def exit_node_skip_prevented(open_list, end):  # method is used to remove visited nodes from the open_list
    for step in open_list:  # plus removing the exit node from the list if it is not the actual target.
        if step.visited:  # if we didn't remove the nodes the algorithm would used the exit node to cheat ;)
            open_list = AStar.clean_open_set(open_list, step)
        if step.x == step.total_row and end.x != end.total_row:
            open_list = AStar.clean_open_set(open_list, step)
    return open_list


def el_beam(open_list, k):  # used to sort open_list by h score
    open_set = set(open_list)
    open_list = list(open_set)
    open_list.sort(key=lambda step: step.h)
    k_best = open_list[:k]
    for move in k_best:
        move.visited = True
    return k_best  # returns k best


def local_k_beam(start, end):  # k-beam algorithm
    k = 3
    final_path = []
    open_list = []
    start.h = AStar.h_score(start, end)  # calculating starting node h score
    start.reset_previous()
    start.visited = True
    found = False
    counter = 0
    if start.h == 0:  # if the starting node is already located in the goal position
        final_path = [start]
        return final_path
    else:
        open_list.extend(start.neighbors)
        open_list = exit_node_skip_prevented(open_list, end)
        update_h_score(open_list, start, end)
    while len(open_list) != 0:
        k_best = el_beam(open_list, k)
        if counter == 0:
            start.consider.append(k_best)  # for printing bag
            counter += 1
        open_list.clear()
        for node in k_best:
            if node.h != 0:
                open_list.extend(node.neighbors)  # add current node (from k-best) neighbors
                open_list = exit_node_skip_prevented(open_list, end)  # filter nodes
                update_h_score(open_list, node, end)
            else:  # when we reached goal
                temp = node
                final_path.clear()
                final_path.append(temp)  # add temp node into the list called final path
                while temp.previous and temp is not start:  # while there is a node that led to the temp node
                    final_path.insert(0, temp.previous)  # insert the previous node into the list
                    temp = temp.previous
                return final_path
    if not found:
        final_path.clear()
    return final_path


def genetic_solve(node_grid, start, end, population_size, mutation_rate):  # genetic algorithm
    life_time = 120
    life_cycle = 1
    pop_counter = 0
    final_path = [start]
    new_population = []
    start.h = AStar.h_score(start, end)  # calculating starting node h score
    if start.h == 0:  # if start node == end node
        return final_path
    else:
        population = initialise_population(start, end, population_size, mutation_rate)
        answer = next_evolutionary_step(node_grid, population, end, life_cycle, life_time)
        found = answer[0]
        if len(answer) == 3:
            life_cycle = answer[2]
        while life_cycle < life_time and not found:
            while pop_counter < population_size:
                child = mutate(reproduction(natural_selection(population, 2)), node_grid)  # producing a child
                if child is not None:
                    new_population.append(child)
                    pop_counter += 1
            population = copy.copy(new_population)
            new_population.clear()
            pop_counter = 0
            life_cycle += 1
            answer = next_evolutionary_step(node_grid, population, end, life_cycle, life_time)  # adding the next step
            found = answer[0]  # and grading all routes in population
            if len(answer) == 3:
                life_cycle = answer[2]
        if not found:
            final_path.clear()
        else:
            final_path = decode_path(answer[1], node_grid, final_path)
            start.route_evolution.append(answer[1])  # for printing parents
        return final_path


def decode_path(route, node_grid, final_path):  # method is used to translate geno-type (directions) into
    # phenotype (nodes)
    route.tail[0] = route.start.x
    route.tail[1] = route.start.y
    for i in range(len(route.dna)):
        route.tail += route.dna[i]
        if route.tail[0] != route.end.total_row and route.tail[1] != route.end.total_row:
            final_path.append(node_grid.get_grid()[route.tail[0]][route.tail[1]])
        else:
            final_path.append(route.end)
    return final_path


def natural_selection(population, number_of_parents):  # method manages selection of parents
    wheel = make_wheel(population)  # creating wheel of fortune (where the route with the higher fitness gets a bigger
    # slot (bigger pizza slice) meaning the probability of choosing the route for mating gets bigger)
    parents = selection(wheel, number_of_parents)
    return parents


def reproduction(parents):  # creating a new born route
    dad = parents[0]
    mom = parents[1]
    child = Route(dad.start, dad.end, dad.mutation_rate, dad.pop_size)
    child.selections.append(dad.selection_probability)
    child.selections.append(mom.selection_probability)
    crossover_index = int(random.uniform(1, len(dad.dna)))  # random index to trim genome
    child.crossover_index = crossover_index
    child.heritage.append(dad.dna)
    child.heritage.append(mom.dna)
    mom_genome = mom.dna[crossover_index:len(mom.dna)]
    child.dna.extend(dad.dna[0:crossover_index])
    child.dna.extend(mom_genome)  # merging both parents dna into the child
    child.update_head()
    if child.head[0] >= child.start.total_row or child.head[1] >= child.start.total_row:  # if after the merge the head
        return None  # is off the board borders > not coherent
    return child


def check_genome(child, node_grid):  # method is responsible of checking if the dna is a coherent genotype
    max_option = 1000
    dna_valid = True
    child.head[0] = child.start.x  # reset head coordinates
    child.head[1] = child.start.y
    for i in range(len(child.dna)):
        if i < child.crossover_index:  # we know the path til the crossover index is coherent
            child.head += child.dna[i]  # updating head coordinates til we reached the crossover index
        else:
            if (child.head[0] >= child.start.total_row or child.head[1] >= child.start.total_row) \
                    or (child.head[0] <= -1 or child.head[1] <= -1):  # if the head cordi is out of the board borders
                return None
            temp = node_grid.get_grid()[child.head[0]][child.head[1]]  # get the actual node from node_grid
            max_option = len(temp.neighbors_direction)  # where the head points to so we know what possible direction-
            for direction in temp.neighbors_direction:  # -we have
                if child.dna[i] in direction[1]:  # if the chromosome were looking at is a possible direction
                    child.head += child.dna[i]  # update head
                    break  # break inner for loop and move on to the next chromosome in the chain
                else:
                    max_option -= 1
        if max_option == 0:  # eventually if condition is met >
            # meaning the chromosome where looking at is not a possible direction
            return False, i  # method return false (meaning dna is not valid) and the index where it crashed
    return dna_valid, 0


def mutate(child, node_grid):  # responsible on mutating children
    if child is not None:
        dna_valid = check_genome(child, node_grid)  # check for coherent dna
        if dna_valid is not None and dna_valid[0]:  # if dna is valid > no mutation is needed
            return child
        elif random.uniform(0, 1) < child.mutation_rate:  # if the dna is not coherent the mutation process will try to
            # fix dna
            if 0 <= child.head[0] < len(node_grid.get_grid()) and 0 <= child.head[1] < len(node_grid.get_grid()):
                mutation_direction = random_step(node_grid.get_grid()[child.head[0]][child.head[1]], child.end)
                child.dna[dna_valid[1]] = mutation_direction
                child.mutated = True
                return child
        else:
            return child
    else:
        return child


def the_one(route, end):  # method is responsible of stopping the program when the target is found
    route.update_head()
    target_found = False
    coherent_path = route.path_legit()  # check for a proper path
    if end.x == end.total_row and route.head[0] == route.end.total_row:  # only if the target is the exit node and the
        # head is located at the last row
        route.head[0] = route.end.total_row  # fix head cordi to >> (6,6)
        route.head[1] = route.end.total_row
    if route.head[0] == end.x and route.head[1] == end.y and coherent_path:  # if we reached the target and the path is
        target_found = True  # coherent > we found a solution
    return target_found, route


def next_evolutionary_step(node_grid, population, end, life_cycle, life_time):  # method is responsible for the update
    # of all members of population plus adding the next step to each and every one of them
    solution = None
    for route in population:
        solution = the_one(route, end)  # first check if we reached the end after populating with new routes
        if solution[0]:  # if we reached the target
            return solution
        route.update_head()
        if 0 <= route.head[0] < len(node_grid.get_grid()) and 0 <= route.head[1] < len(node_grid.get_grid()):
            head_node = node_grid.get_grid()[route.head[0]][route.head[1]]  # if the head cordi is within the
            direction_dna = random_step(head_node, end)  # board borders > generate move from list of possible moves
        else:  # from that spot in the grid , other wise generate a random move from start list of directions
            direction_dna = random_step(route.start, end)
        if direction_dna is not None:
            route.dna.append(direction_dna)  # add the next chromosome in the chain
            route.head += direction_dna
            solution = the_one(route, end)
            if solution[0]:  # if we reached the target
                return solution
            route.update_head()
            if 0 <= route.head[0] < len(node_grid.get_grid()) and 0 <= route.head[1] < len(node_grid.get_grid()):
                route.current = node_grid.get_grid()[route.head[0]][route.head[1]]  # if the head cordi is within the
            else:  # board borders > update current
                route.current = None
            route.fit = fitness(route)  # update fitness of route
            route.generation = life_cycle
        else:
            life_cycle = life_time
            return False, None, life_cycle
    return solution


def blocked(copied_list_of_neighbor_direction):  # check for an agent being blocked from all sides (part of random_step)
    block = True
    for neighbor in copied_list_of_neighbor_direction:
        if neighbor[0].my_state != 1:
            block = False
    return block


def random_step(current_node, end):  # generate random geno- type step
    copied_list_of_neighbor_direction = copy.copy(current_node.neighbors_direction)
    for node_relative_direction in copied_list_of_neighbor_direction:
        if node_relative_direction[0].x == node_relative_direction[0].get_total_row() and \
                end.x != node_relative_direction[0].get_total_row():
            copied_list_of_neighbor_direction.remove(node_relative_direction)
    if len(copied_list_of_neighbor_direction) > 0 and not blocked(copied_list_of_neighbor_direction):
        move = random.choice(copied_list_of_neighbor_direction)
        direction_dna = move[1]
    else:
        direction_dna = None
    return direction_dna


def initialise_population(start, end, population_size, mutation_rate):  # creating the first pop
    population = list(range(population_size))
    for i in range(population_size):
        population[i] = Route(start, end, mutation_rate, population_size)
    return population


def fitness(route):  # method is responsible of grading all route in population
    fit = 0
    sum_of_vectors = np.array([0, 0])
    if route.current is not None:  # if the current node is none meaning when we tried to update it using head cordi,
        distance = AStar.h_score(route.current, route.end)  # the head cordi was out of the board borders >> fitness = 0
        if distance >= 1:  # basic grade based on h score
            fit = 1 / distance
            fit = fit ** 2  # to the power of 2 to space out the different type of fit scores
            if route.current.my_state == 1:
                fit = fit * 0.00000001
            coherent_path = route.path_legit()
            if coherent_path:  # we want to encourage good paths
                fit * 10
            else:
                fit = fit * 0.00000001  # and discourage bad ones
            if len(route.dna) > 3:
                last_few = len(route.dna) - 2
                list_to_average = route.dna[last_few:]
                for i in range(len(list_to_average)):
                    sum_of_vectors += list_to_average[i]
                s = np.sum(sum_of_vectors)
                if s == 0:  # meaning we are walking in circles > fitness takes a hit
                    fit = fit * 0.0000042
                elif -1 <= s <= 1:
                    fit = fit * 0.00042
                elif -2 <= s <= 2:
                    fit = fit * 2  # encourage moving and not staying in the same place
        else:
            route.hit_target = True
    return fit


def make_wheel(population):  # creating wheel of fortune
    wheel = []
    total = sum(fitness(p) for p in population)
    top = 0
    for p in population:
        f = fitness(p) / total
        wheel.append((top, top + f, p))
        top += f
    return wheel


def bin_search(wheel, num):  # finding the parent using binary search
    mid = len(wheel) // 2
    low, high, answer = wheel[mid]
    if low <= num <= high:
        return answer
    elif high < num:
        return bin_search(wheel[mid + 1:], num)
    else:
        return bin_search(wheel[:mid], num)


def selection(wheel, n):  # method is responsible of selecting and returning 2 parents from wheel
    step_size = 1.0 / n
    answer = []
    r = random.random()
    route = bin_search(wheel, r)
    route.selection_probability = round(r, 3)
    answer.append(route)
    while len(answer) < n:
        r += step_size
        if r > 1:
            r %= 1
        route = bin_search(wheel, r)
        route.selection_probability = round(r, 3)
        answer.append(route)
    return answer


def algorithm_name(search_method):  # for progress bar aesthetics purposes
    if search_method == 1:
        return 'A*'
    if search_method == 2:
        return 'Hill Climbing'
    if search_method == 3:
        return 'Simulated Annealing'
    if search_method == 4:
        return 'Local Beam'
    if search_method == 5:
        return 'Genetic Algorithm'


def multi_agent_handler(node_grid, detail_output, search_method):
    g = nx.Graph()  # creating a graph object from the networkx library > the aim is to create a bipartite graph
    # where one side represent the agents and the other the targets
    agents = node_grid.get_agents()
    targets = node_grid.get_targets()
    shortest_paths = [[[]] * len(targets) for _ in range(len(agents))]  # creating a matrix where
    # shortest_paths[i][j] = shortest path from agent i to target j
    for k in range(len(agents)):  # this for loop is responsible of adding enough nodes (networkx type nodes)
        # representing agents and targets
        g.add_node(k)  # agent node (numbered: 0..1..2..3..4)
        g.add_node(len(agents) + k)  # target node (numbered: 5..6..7..8..9)
    for i in tqdm.tqdm(range(len(agents)), 'Calculating ' + str(len(agents)) + ' paths. . . Algorithm being used->> ' +
                                           algorithm_name(search_method), len(agents)):
        for j in range(len(targets)):
            if search_method == 1:  # calling the start path function in A-star
                # object that output's the optimal path from agent i to target j
                shortest_paths[i][j] = AStar.start_path(agents[i], targets[j])
            if search_method == 2:
                shortest_paths[i][j] = hill_climb(agents[i], targets[j])
            if search_method == 3:
                shortest_paths[i][j] = simulated_annealing(agents[i], targets[j])
            if search_method == 4:
                shortest_paths[i][j] = local_k_beam(agents[i], targets[j])
                reset_visiting(node_grid)
            if search_method == 5:
                shortest_paths[i][j] = genetic_solve(node_grid, agents[i], targets[j], 10, 0.24)
            if len(shortest_paths[i][j]) > 0:  # meaning there is actually a path from that agent to that target
                g.add_weighted_edges_from([(i, len(agents) + j, 1 / len(shortest_paths[i][j]))])  # add a weighted edge
                # between the two nodes in g (I choose the weight to be 1/len() because the function that I found knows
                # how to find a maximum weight match and by doing so I made sure that the shortest path of the
                # maximum weight)
    min_match = nx.algorithms.matching.max_weight_matching(g)  # black box returning the minimum match between
    # agents and targets
    print_solution(min_match, shortest_paths, node_grid, detail_output,
                   search_method)  # method responsible of printing solution


def printing_assistance(board, path, i, h, goal_board_number, search_method, consider_list):
    start = i - 1
    while i < len(path) and path[i].get_state() == 2:  # while the next step is occupied by a agent
        i += 1
    finish = i + 1
    while i > start:
        path[i - 1].set_state(0)  # making the step
        if path[i].get_x() != len(board.get_grid()) and path[i].get_y() != len(
                board.get_grid()):  # if the step we are
            # about to make is not a step towards a exit node (exit node are the way to disappear from the board
            # and we want to keep their state as 0)
            path[i].set_state(2)  # making the step
        board.print_board(h, goal_board_number, search_method, consider_list, False, 0, 0, '')  # print the board after
        i -= 1
    return finish


def update_consideration(path, consider_list):  # responsible of updating the consideration to
    i = 1  # the actual one being printed first
    while i < len(path) and path[i].get_state() == 2:  # while the next step is occupied by a agent
        i += 1
    if len(consider_list) > 0 and len(path) > 1:
        init_move = consider_list.pop(-1)
        if path[i].x == path[i].total_row:
            actual_move = (path[i - 1], 'out', init_move[2])
            consider_list.append(actual_move)
        else:
            actual_move = (path[i - 1], path[i], init_move[2])
            consider_list.append(actual_move)


def remove_dup(consider_list, about_to_be_removed):  # used in about_to_print
    for move in about_to_be_removed:
        consider_list = AStar.clean_open_set(consider_list, move)
    return consider_list


def about_to_print(optimal_matching, node_grid, paths, search_method, k):  # method returns the bag of consideration
    # that will be printed first
    consider_list = None
    moves_in_bag = 0
    bag = []
    for match in optimal_matching:
        agent = min(match)
        target = max(match) - len(node_grid.get_agents())
        if len(paths[agent][target]) > 0:  # gets the first match with an actual path
            consider_list = node_grid.get_agents()[agent].consider[target]
            about_to_be_removed = []
            first_one_to_move = node_grid.get_agents()[agent]
            to_target = node_grid.get_targets()[target]
            path = paths[agent][target]
            if search_method == 4:
                for node in consider_list:
                    if node.my_state == 2:
                        about_to_be_removed.append(node)
                consider_list = remove_dup(consider_list, about_to_be_removed)
                moves_in_bag += len(consider_list)
                bag.append((consider_list, first_one_to_move, to_target))
                if moves_in_bag < k:
                    continue
                else:
                    return bag
            if search_method == 3:
                update_consideration(path, consider_list)
                break
        else:
            continue
    if search_method == 4:
        return bag
    return consider_list


def print_bag(node_grid, about_to, h, goal_board_number, search_method, k):  # responsible off managing the bag of
    size = len(node_grid.get_grid())    # consideration printing
    consider_list = []
    counter = 0
    for consideration in about_to:
        if len(consideration[0]) == 0:
            continue
        for fake_step in consideration[0]:
            fake_board = copy.deepcopy(node_grid)
            fake_board.get_grid()[consideration[1].x][consideration[1].y].set_state(0)
            if fake_step.x != size and fake_step.y != size:
                fake_board.get_grid()[fake_step.x][fake_step.y].set_state(2)
            fake_board.print_board(h, goal_board_number, search_method, consider_list, True, counter, 0, 0, '')  # print
            counter += 1
            if counter == k:
                break


def print_parents(optimal_matching, node_grid, paths):  # responsible off managing the parents printing
    mom_grid = copy.deepcopy(node_grid)  # making copies of the grid for a one time print
    child_grid = copy.deepcopy(node_grid)
    for match in optimal_matching:
        agent = min(match)
        target = max(match) - len(node_grid.get_agents())
        if len(paths[agent][target]) > 0:  # gets the first match with a path
            route = node_grid.get_agents()[agent].route_evolution[target]
            final_parent = [route.start]
            dad_dna = route.heritage[0]
            mom_dna = route.heritage[1]
            my_dna = route.dna
            route.dna = dad_dna
            final_parent = decode_path(route, node_grid, final_parent)
            set_state(node_grid, final_parent, 1, route.crossover_index)
            node_grid.print_board(1, 100, 5, final_parent, True, 1, route.selections[0], '')  # printing dad
            final_parent.clear()
            final_parent = [route.start]
            route.dna = mom_dna
            final_parent = decode_path(route, mom_grid, final_parent)
            set_state(mom_grid, final_parent, 2, route.crossover_index)
            mom_grid.print_board(1, 100, 5, final_parent, True, 2, route.selections[1], '')  # printing mom
            final_parent.clear()
            final_parent = [route.start]
            route.dna = my_dna
            final_parent = decode_path(route, child_grid, final_parent)
            set_state(child_grid, final_parent, 3, route.crossover_index)
            if route.mutated:  # printing child
                child_grid.print_board(1, 1, 5, final_parent, True, 3, 0, ' Yes ')
            else:
                child_grid.print_board(1, 1, 5, final_parent, True, 3, 0, ' No ')
            break


def set_state(node_grid, final_dad, family_member, crossover_index):  # insert values into copies of node_grid
    counter = 0  # in order to easily see genetic creations
    for node in final_dad:
        if node.x != node.total_row and node.y != node.total_row:
            if family_member == 1:
                node_grid.get_grid()[node.x][node.y].my_state = 3
            elif family_member == 2:
                node_grid.get_grid()[node.x][node.y].my_state = 4
            elif family_member == 3:
                if counter < crossover_index:
                    node_grid.get_grid()[node.x][node.y].my_state = 3
                    counter += 1
                else:
                    node_grid.get_grid()[node.x][node.y].my_state = 4


def print_solution(optimal_matching, paths, node_grid, detail_output, search_method):  # handles solution print
    h = 0
    parents_board = copy.deepcopy(node_grid)
    goal_board_number = 1
    consider_list = []
    if detail_output and len(node_grid.get_agents()) <= len(optimal_matching) and search_method == 3:
        consider_list = about_to_print(optimal_matching, node_grid, paths, search_method, 3)
        h = 1
    node_grid.print_board(h, goal_board_number, search_method, consider_list, False, 0, 0, '')  # print starting board
    if detail_output and len(node_grid.get_agents()) <= len(optimal_matching) and search_method == 4:
        consider_list = about_to_print(optimal_matching, node_grid, paths, search_method, 3)
        h = 1
        print_bag(node_grid, consider_list, h, goal_board_number, search_method, 3)
    if len(node_grid.get_agents()) > len(optimal_matching):  # if agents list is bigger then the optimal match meaning
        # that there is at least one agent blocked from getting to his target >> no solution
        print('There is no possible solution..')
    else:
        for match in optimal_matching:  # iterate over all matches
            agent = min(match)
            target = max(match)
            h += AStar.h_score(node_grid.get_agents()[agent],
                               node_grid.get_targets()[target - len(node_grid.get_agents())])
            # adding the h score from the agent[i] to the target[j]
            goal_board_number += len(paths[agent][target - len(node_grid.get_agents())]) - 1
        if not detail_output:  # if the h value is wanted in the output
            h = -1  # in case the h value is not needed
        for match in optimal_matching:
            agent = min(match)
            target = max(match)
            current_path = paths[agent][target - len(node_grid.get_agents())]  # gets the shortest path from
            # agent[i] to the target[j]
            length_of_path = len(current_path)
            if length_of_path != 1:  # if the length_of_path is equal to 1 > meaning the node is already in the desired
                # target and so we don't need to print the board again if no move has been made
                i = 1  # starting in 1 because its actually the first step in the path (0 is the starting node of
                # the path)
                while i < length_of_path:
                    if current_path[i].get_state() == 2:  # if the step we are about to take is occupied by a agent
                        i = printing_assistance(node_grid, current_path, i, h, goal_board_number,
                                                search_method, consider_list)  # call for assistance
                    else:  # meaning the next step is not occupied by a agent
                        current_path[i - 1].set_state(0)  # making the step
                        if current_path[i].get_x() != len(node_grid.get_grid()) or current_path[i].get_y() != len(
                                node_grid.get_grid()):  # if the step we are about to make is not a step towards a exit
                            # node (exit node are the way to disappear from the board and we want
                            # to keep their state as 0)
                            current_path[i].set_state(2)  # making the step
                        node_grid.print_board(h, goal_board_number, search_method, consider_list, False, 0, 0, '')
                        # print board after changes
                        i += 1
        if search_method == 5 and detail_output:
            print_parents(optimal_matching, parents_board, paths)


def main():
    starting_board = [[2, 0, 2, 0, 2, 0],
                      [0, 0, 0, 2, 1, 2],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0],
                      [2, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]]
    goal_board = [[2, 0, 2, 0, 0, 0],
                  [0, 0, 0, 2, 1, 2],
                  [1, 0, 0, 0, 0, 2],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]]
    find_path(starting_board, goal_board, 5, True)


if __name__ == '__main__':
    print('Run Directly')
    main()
else:
    print("Run From Import")
