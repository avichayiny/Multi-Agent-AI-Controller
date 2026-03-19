import ex1_check
import search
import utils
id = ["206626947"]



class WateringProblem(search.Problem):
    """This class implements a pressure plate problem"""


    def __init__(self, initial):
        
        size = initial["Size"]
        walls = frozenset(initial["Walls"])
        taps = tuple(sorted(initial["Taps"].items()))
        plants = tuple(sorted(initial["Plants"].items()))
        robots = tuple(sorted(initial["Robots"].items()))
        initial_state = ( taps, plants, robots)

        rows, cols = size
        targets = list(initial["Taps"].keys()) + list(initial["Plants"].keys())
        self.landmarks_data = compute_distances(rows, cols, targets, walls)

        self.h_cache = {}

        self.plant_map_index = {loc: i for i, (loc, amt) in enumerate(plants)}
        self.tap_map_index = {loc: i for i, (loc, amt) in enumerate(taps)}
        self.valid_moves = precompute_valid_moves(rows, cols, walls)
        self.max_global_capacity = 1
        if initial["Robots"]:
            self.max_global_capacity = max(r[3] for r in initial["Robots"].values())
        

        self.global_min_segment = float('inf')
        
        tap_locs = [t[0] for t in taps]
        plant_locs = [p[0] for p in plants]
        
        for t_loc in tap_locs:
            if t_loc in self.landmarks_data:
                for p_loc in plant_locs:
                    dist = self.landmarks_data[t_loc][p_loc[0]][p_loc[1]]
                    if dist < self.global_min_segment:
                        self.global_min_segment = dist
        
       
        if self.global_min_segment == float('inf'):
            self.global_min_segment = 0
        """ Constructor only needs the initial state.
        Don't forget to set the goal or implement the goal test"""
        search.Problem.__init__(self, initial_state)


    def successor(self, state):
        successors = []
        taps, plants, robots = state 
        
        
        is_single_player = (len(robots) == 1)
        
        if is_single_player:
            occupied_cells = set() 
        else:
            occupied_cells = { (r[1][0], r[1][1]) for r in robots }

        for i, (r_id, (r_r, r_c, r_load, r_cap)) in enumerate(robots):
            curr_loc = (r_r, r_c)
            
            is_isolated = True
            if not is_single_player:
                for j, (_, (o_r, o_c, _, _)) in enumerate(robots):
                    if i == j: continue 
                    dist = abs(r_r - o_r) + abs(r_c - o_c)
                    if dist <= 2: 
                        is_isolated = False
                        break
            
            safe_to_prune = is_single_player or is_isolated

            
            if curr_loc in self.plant_map_index:
                plant_idx = self.plant_map_index[curr_loc]
                p_loc, p_amt = plants[plant_idx]
                
                if p_amt > 0 and r_load > 0:
                    new_robot = (r_id, (r_r, r_c, r_load - 1, r_cap))
                    new_robots = robots[:i] + (new_robot,) + robots[i+1:]
                    new_plant = (curr_loc, p_amt - 1)
                    new_plants = plants[:plant_idx] + (new_plant,) + plants[plant_idx+1:]
                    
                    pour_action = (f"POUR{{{r_id}}}", (taps, new_plants, new_robots))
                    
                    if all(p[1] == 0 for p in new_plants):
                         return [pour_action]

                    if p_amt == r_load:
                        total_needed = sum(p[1] for p in plants)
                        if total_needed == p_amt:
                            return [pour_action]
                    
                    if safe_to_prune:
                        successors.append(pour_action)
                        continue 
                    
                    successors.append(pour_action)

            if curr_loc in self.tap_map_index:
                tap_idx = self.tap_map_index[curr_loc]
                t_loc, t_amt = taps[tap_idx]
                
                if t_amt > 0 and r_load < r_cap:
                    new_robot = (r_id, (r_r, r_c, r_load + 1, r_cap))
                    new_robots = robots[:i] + (new_robot,) + robots[i+1:]
                    new_tap = (curr_loc, t_amt - 1)
                    new_taps = taps[:tap_idx] + (new_tap,) + taps[tap_idx+1:]
                    
                    load_action = (f"LOAD{{{r_id}}}", (new_taps, plants, new_robots))
                    
                    if is_single_player:
                        total_needed = sum(p[1] for p in plants)
                        if r_load < total_needed:
                            successors.append(load_action)
                            continue

                    successors.append(load_action)

            possible_moves = self.valid_moves.get(curr_loc, [])
            for action_name, new_loc in possible_moves:
                if new_loc not in occupied_cells:
                    new_robot = (r_id, (new_loc[0], new_loc[1], r_load, r_cap))
                    new_robots = robots[:i] + (new_robot,) + robots[i+1:]
                    
                    successors.append((f"{action_name}{{{r_id}}}", (taps, plants, new_robots)))

        return successors


    def goal_test(self, state):
        _, plants, _ = state
        for _, amount in plants:
            if amount > 0:
                return False
        return True
        utils.raiseNotDefined()


    def h_astar(self, node):
        state = node.state
        taps, plants, robots = state
        if state in self.h_cache:
            return self.h_cache[state]

        total_needed = 0
        thirsty_plants = [] 
        for loc, amt in plants:
            total_needed += amt
            if amt > 0:
                thirsty_plants.append(loc)
        if total_needed == 0:
            return 0

        current_water_in_robots = sum(r[1][2] for r in robots)
        
        must_pour = total_needed
        must_load = max(0, total_needed - current_water_in_robots)
        base_actions_cost = must_pour + must_load


        active_taps = [loc for loc, amt in taps if amt > 0]

        min_dist_tap_to_plant = self.global_min_segment

        first_trip_cost = 0


        if current_water_in_robots > 0:
            ld = self.landmarks_data
            min_dist_loaded = float('inf')

            if thirsty_plants:
                for _, (r_r, r_c, r_load, _) in robots:
                    if r_load > 0:
                        best = min_dist_loaded 

                        for p in thirsty_plants:
                            d = ld[p][r_r][r_c]
                            if d < best:
                                best = d
                                if best == 0:   
                                    break

                        if best < min_dist_loaded:
                            min_dist_loaded = best

            
            first_trip_cost = min_dist_loaded if min_dist_loaded != float('inf') else 0

        else:

            
        
            min_dist_robot_to_tap = float('inf')
            ld = self.landmarks_data 
            active = active_taps 

            if active:
                for _, (r_r, r_c, _, _) in robots:
                    best = 999999
                    for t in active:
                        d = ld[t][r_r][r_c]
                        if d < best:
                            best = d
                        if best == 0:
                            break
                    if best < min_dist_robot_to_tap:
                        min_dist_robot_to_tap = best

            
            if min_dist_robot_to_tap == float('inf'): min_dist_robot_to_tap = 0

            first_trip_cost = min_dist_robot_to_tap + min_dist_tap_to_plant

        total_trips = search.math.ceil(total_needed / self.max_global_capacity)
        
        num_excess_trips = max(0, total_trips - len(robots))

        refill_cost = 0
        if num_excess_trips > 0:
            refill_cost = num_excess_trips * (min_dist_tap_to_plant * 2)

        num_first_wave_trips = min(total_trips, len(robots))
        approach_cost = num_first_wave_trips * first_trip_cost

        self.h_cache[state] = base_actions_cost + approach_cost + refill_cost
        return base_actions_cost + approach_cost + refill_cost

    
        utils.raiseNotDefined()

    

    def h_gbfs(self, node):
        """ This is the heuristic. It gets a node (not a state)
        and returns a goal distance estimate"""
        state = node.state
        taps, plants, robots = state 
    
        total_needed = sum(amount for _, amount in plants)
        if total_needed == 0:
            return 0
    
        base_score = total_needed * 1000
    
        active_taps = [loc for loc, amt in taps if amt > 0]
        thirsty_plants = [loc for loc, amt in plants if amt > 0]
    
    
        if not thirsty_plants:
            return 0
    
        best_robot_score = float('inf')
    
        for _, (r_r, r_c, r_load, _) in robots:
            r_loc = (r_r, r_c)
        
            if r_load > 0:
                min_dist_to_plant = min(
                    abs(r_loc[0] - p[0]) + abs(r_loc[1] - p[1])
                    for p in thirsty_plants
                )
                current_score = min_dist_to_plant
            
            else:
                if active_taps:
                    best_tap_dist = float('inf')
                    chosen_tap = None
                
                    for t in active_taps:
                        d = abs(r_loc[0] - t[0]) + abs(r_loc[1] - t[1])
                        if d < best_tap_dist:
                            best_tap_dist = d
                            chosen_tap = t

                    dist_tap_to_plant = min(
                        abs(chosen_tap[0] - p[0]) + abs(chosen_tap[1] - p[1])
                        for p in thirsty_plants
                    )
                
                    current_score = best_tap_dist + dist_tap_to_plant
                else:
                    current_score = float('inf')

            if current_score < best_robot_score:
                best_robot_score = current_score

        if best_robot_score == float('inf'):
            best_robot_score = 500

        return base_score + best_robot_score
        utils.raiseNotDefined()


def compute_distances(rows, cols, targets, walls):
    distances = {}
    unique_targets = set(targets)
    for landmark in unique_targets:
        distances[landmark] = compute_distances_from_landmark(landmark, rows, cols, walls)
    return distances

def compute_distances_from_landmark(start_node, rows, cols, walls):
    distances = [[float('inf') for _ in range(cols)] for _ in range(rows)]
    
    queue = utils.FIFOQueue()
    queue.append((start_node, 0)) 
    
    visited = set()
    visited.add(start_node)
    
    start_x, start_y = start_node
    distances[start_x][start_y] = 0
    
    while queue:
        try:
            (cx, cy), dist = queue.pop()
        except IndexError:
            break

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in walls and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    
                    distances[nx][ny] = dist + 1
                    
                    queue.append(((nx, ny), dist + 1))
    
    return distances

def precompute_valid_moves(rows, cols, walls):
    valid_moves = {}
    
    directions = [
        ("UP", -1, 0),
        ("DOWN", 1, 0),
        ("RIGHT", 0, 1),
        ("LEFT", 0, -1)
    ]
    
    for r in range(rows):
        for c in range(cols):
            curr_loc = (r, c)
            
            if curr_loc in walls:
                continue
            
            legal_moves_from_here = []
            
            for action_name, dr, dc in directions:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    if neighbor not in walls:
                        legal_moves_from_here.append((action_name, neighbor))
            
            if legal_moves_from_here:
                valid_moves[curr_loc] = legal_moves_from_here
                
    return valid_moves


def create_watering_problem(game):
    print("<<create_watering_problem")
    """ Create a pressure plate problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return WateringProblem(game)


if __name__ == '__main__':
    ex1_check.main()
