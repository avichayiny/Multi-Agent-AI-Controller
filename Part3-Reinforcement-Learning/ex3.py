import ext_plant
import collections
import math

id = ["206626947"]
"""
Using AI:
The design, planning, and idea behind the code were mine. 
I used the implementation itself in Gemini. 
Each time I wanted to write a function or snippet, I asked it to implement it for me in a general way. 
I examined every implementation and made sure I was happy with it. 
If necessary (every time to be precise), I added precisions to the code myself, appropriate mathematical calculations, or more specific edge cases and resolutions of the code.
I also used AI to debug my code and find where it doesn't work, and where there are gaps.
"""
class Controller:
    def __init__(self, game: ext_plant.Game):
        self.game = game
        self.model = game.get_problem() 
        self.rows, self.cols = self.model["Size"]
        self.walls = set(self.model["Walls"])
        self.robot_capacities = {rid: r_data[3] for rid, r_data in self.model["Robots"].items()}
        
        self.probs = {} 
        self.robot_stats = {}
        for rid in self.robot_capacities:
            self.probs[rid] = 1.0
            self.robot_stats[rid] = [1.0, 1.0] 
            
        self.plant_values = {}
        self.plant_stats = {}
        
        max_rewards = self.game.get_plants_max_reward()
        for pos, max_r in max_rewards.items():
            self.plant_values[pos] = float(max_r)
            self.plant_stats[pos] = [float(max_r), 1.0]

        self.initial_plant_needs = {pos: amt for pos, amt in self.model["Plants"].items()}
        self.goal_reward = self.model["goal_reward"]
        
        self.landmarks_dist = {}
        targets = list(self.model["Taps"].keys()) + list(self.model["Plants"].keys())
        for r_data in self.model["Robots"].values():
            targets.append((r_data[0], r_data[1]))
            
        for target in set(targets):
            self.landmarks_dist[target] = self.run_bfs(target)
            
        self.taps_locations = list(self.model['Taps'].keys())
        
        
        raw_initial = game.get_current_state()
        self.internal_initial_state = self.make_state_hashable(raw_initial) 
        self.raw_initial_state = raw_initial
        
        
        self.prev_state = None
        self.prev_action = None 
        self.memo = {}
        self.GAMMA = 0.98

    def make_state_hashable(self, state):
        robots, plants, taps, total_need = state
        robots_tuple = tuple((r[0], r[1], r[2]) for r in robots)
        plants_tuple = tuple(plants)
        taps_tuple = tuple(taps)
        return (robots_tuple, plants_tuple, taps_tuple, total_need)

    
    def update_model(self, current_state):
        """מעדכן את self.probs ואת self.plant_values על סמך הצעד האחרון"""
        if self.prev_state is None or self.prev_action is None:
            return

        action_type, rid = self.prev_action
        if action_type == "RESET":
            return

        
        robots_prev, plants_prev, _, _ = self.prev_state
        robots_curr, plants_curr, _, _ = current_state
        
        r_prev = next(r for r in robots_prev if r[0] == rid)
        r_curr = next(r for r in robots_curr if r[0] == rid)
        
        is_success = False
        
        if action_type in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[action_type]
            wanted_pos = (r_prev[1][0] + dr, r_prev[1][1] + dc)
            if r_curr[1] == wanted_pos:
                is_success = True
                
        elif action_type == "LOAD":
            if r_curr[2] > r_prev[2]:
                is_success = True
                
        elif action_type == "POUR":
            plant_pos = r_prev[1]
            prev_need = dict(plants_prev).get(plant_pos, 0)
            curr_need = dict(plants_curr).get(plant_pos, 0)
            

            if curr_need < prev_need:
                is_success = True

        self.robot_stats[rid][1] += 1 
        if is_success:
            self.robot_stats[rid][0] += 1 
        
 
        self.probs[rid] = max(0.05, self.robot_stats[rid][0] / self.robot_stats[rid][1])


        last_reward = self.game.get_last_gained_reward()
        if action_type == "POUR" and last_reward > 0:
            plant_pos = r_prev[1]
            if plant_pos in self.plant_stats:
                self.plant_stats[plant_pos][0] += last_reward
                self.plant_stats[plant_pos][1] += 1
                

                self.plant_values[plant_pos] = self.plant_stats[plant_pos][0] / self.plant_stats[plant_pos][1]

    def choose_next_action(self, state):
 
        self.update_model(state)
        
        robots, plants, taps, total_need = state
        if total_need == 0: 
            self.prev_action = ("RESET", None)
            self.prev_state = state
            return "RESET"
        
        self.memo = {}
        real_steps_left = self.game.get_max_steps() - self.game.get_current_steps()
        
        
        planning_horizon = real_steps_left
        SEARCH_DEPTH = 8 
        

        val_reset = self.maximize_value(
            self.raw_initial_state, 
            depth=SEARCH_DEPTH, 
            current_horizon=real_steps_left - 1, 
            action="RESET", 
            robot_id=None
        ) - 2.0 
        

        best_val = val_reset
        best_action_str = "RESET"
        best_action_details = ("RESET", None)

        valid_actions = self.get_sorted_valid_actions(state)
        
        if valid_actions:
            for action_name, rid in valid_actions:
                current_q_val = 0
                transitions = self.get_transitions(state, action_name, rid)
                
                
                if action_name == 'POUR':

                    self.prev_state = state
                    self.prev_action = (action_name, rid)
                    return f"{action_name} ({rid})"

                for prob, next_state, reward in transitions:

                    v_next = self.maximize_value(next_state, SEARCH_DEPTH - 1, planning_horizon - 1, action_name, rid)
                    

                    adjusted_reward = reward * 3 if reward > 0 else reward
                    
                    current_q_val += prob * (adjusted_reward + (self.GAMMA * v_next))
                
                if current_q_val > best_val:
                    best_val = current_q_val
                    best_action_str = f"{action_name} ({rid})"
                    best_action_details = (action_name, rid)

        
        self.prev_state = state
        self.prev_action = best_action_details
        
        return best_action_str

    def maximize_value(self, state, depth, current_horizon, action, robot_id):
        if depth == 0 or current_horizon <= 0:
            return self.calculate_greedy_heuristic(state, current_horizon)
        
        state_key = self.make_state_hashable(state)
        memo_key = (state_key, depth, current_horizon)
        
        if memo_key in self.memo:
            return self.memo[memo_key]

        robots, plants, taps, total_need = state
        
        if total_need == 0:

            res = self.goal_reward * 2 + (self.GAMMA * self.calculate_greedy_heuristic(self.raw_initial_state, current_horizon - 1))
            self.memo[memo_key] = res
            return res

        best_v = -float('inf')
        actions = self.get_sorted_valid_actions(state)
        
        if not actions: 
            res = self.calculate_greedy_heuristic(state, current_horizon) 
            self.memo[memo_key] = res
            return res

        for action_name, rid in actions:
            expected_val = 0
            transitions = self.get_transitions(state, action_name, rid)
            
            for prob, next_state, reward in transitions:
                v_next = self.maximize_value(next_state, depth - 1, current_horizon - 1, action_name, rid)
                
                shaped_reward = reward
                if action_name == "POUR" and reward > 0:
                    shaped_reward = reward * 3 
                    
                if next_state[3] == 0:
                    shaped_reward += 50.0
                            
                expected_val += prob * (shaped_reward + (self.GAMMA * v_next))
            
            if expected_val > best_v:
                best_v = expected_val
        
        self.memo[memo_key] = best_v
        return best_v

    def calculate_greedy_heuristic(self, state, horizon):
        if horizon <= 0: return 0
        
        robots, plants, taps, total_need = state
        if total_need == 0:
            return self.goal_reward + (horizon * 1.5)

        sim_plants = {p[0]: p[1] for p in plants if p[1] > 0}
        sim_horizon = horizon
        potential_reward = 0
        
        current_sim_time = 0 
        sim_robots = []
        for rid, (r, c), load in robots:
            sim_robots.append([rid, r, c, load, self.robot_capacities[rid]])

        
        while sim_horizon > 0 and sim_plants:
            best_move = None
            max_score = -float('inf')
            
            for i, r_data in enumerate(sim_robots):
                rid, rr, rc, r_load, r_cap = r_data
                
                
                prob = self.probs[rid]
                time_factor = 1.0 / prob 


                if r_load > 0:
                    for p_pos, p_need in sim_plants.items():
                        dist = self.get_safe_dist(p_pos, rr, rc)
                        real_cost = (dist + 1) * time_factor
                        
                        if real_cost <= sim_horizon:
                            amount = min(r_load, p_need)
                            avg_val = self.plant_values[p_pos]
                            arrival_time = current_sim_time + real_cost
                            discounted_reward = amount * avg_val * (self.GAMMA ** arrival_time)
                            score = discounted_reward / real_cost
                            
                            if score > max_score:
                                max_score = score
                                best_move = (i, real_cost, amount, p_pos, r_load - amount, p_pos, discounted_reward)

                if r_load < r_cap:
                    tap_dist = float('inf'); tap_pos = None
                    for t_pos in self.taps_locations:
                        d = self.get_safe_dist(t_pos, rr, rc)
                        if d < tap_dist: tap_dist = d; tap_pos = t_pos
                    
                    if tap_pos:
                        dist_to_tap = self.get_safe_dist(tap_pos, rr, rc)
                        cost_travel = dist_to_tap * time_factor
                        
                        for p_pos, p_need in sim_plants.items():
                            needed_total = min(r_cap, p_need)
                            if r_load >= needed_total: continue 

                            needed_to_fill = needed_total - r_load
                            cost_loading = needed_to_fill * time_factor
                            cost_to_ready = cost_travel + cost_loading + 1 
                            dist_to_plant = self.get_safe_dist(p_pos, tap_pos[0], tap_pos[1])
                            cost_plant_travel = (dist_to_plant + 1) * time_factor
                            total_real_cost = cost_to_ready + cost_plant_travel
                            
                            if total_real_cost <= sim_horizon:
                                avg_val = self.plant_values[p_pos]
                                arrival_time = current_sim_time + total_real_cost
                                discounted_reward = needed_total * avg_val * (self.GAMMA ** arrival_time)
                                score = discounted_reward / total_real_cost
                                
                                if score > max_score:
                                    max_score = score
                                    best_move = (i, total_real_cost, needed_total, p_pos, 0, p_pos, discounted_reward)

            if best_move:
                idx, cost, amount, new_pos, new_load, p_pos, discounted_reward = best_move
                sim_horizon -= cost
                current_sim_time += cost
                potential_reward += discounted_reward
                
                sim_robots[idx][1] = new_pos[0]
                sim_robots[idx][2] = new_pos[1]
                sim_robots[idx][3] = new_load
                
                if sim_plants[p_pos] <= amount:
                    del sim_plants[p_pos]
                else:
                    sim_plants[p_pos] -= amount
            else:
                break
            if not sim_plants:
                    potential_reward += self.goal_reward
                    potential_reward += (sim_horizon * 1.5)
        return potential_reward

    def get_sorted_valid_actions(self, state):
        robots, plants, taps, total_need = state
        valid_actions = []
        plant_locs = {p[0]: p[1] for p in plants}
        tap_amounts = {t[0]: t[1] for t in taps}
        
        for rid, (rr, rc), load in robots:
            capacity = self.robot_capacities[rid]
            if (rr, rc) in plant_locs and plant_locs[(rr, rc)] > 0 and load > 0:
                 return [("POUR", rid)]
            
            if (rr, rc) in tap_amounts and tap_amounts[(rr, rc)] > 0 and load == 0:
                return [("LOAD", rid)]
            
            current_robot_actions = []
            if (rr, rc) in tap_amounts and tap_amounts[(rr, rc)] > 0 and load < capacity:
                current_robot_actions.append(("LOAD", rid))
            
            possible_moves = []
            for move in ["UP", "DOWN", "LEFT", "RIGHT"]:
                dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[move]
                nr, nc = rr + dr, rc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
                    is_occupied = False
                    for other_rid, (orr, orc), _ in robots:
                        if other_rid != rid and (orr, orc) == (nr, nc):
                            is_occupied = True; break
                    if not is_occupied:
                        possible_moves.append(move)
            
            targets = []
            if load > 0: targets = [p for p, amt in plant_locs.items() if amt > 0]
            elif load < capacity: targets = [t for t, amt in tap_amounts.items() if amt > 0]
            
            def move_score(move_name):
                dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[move_name]
                nr, nc = rr + dr, rc + dc
                min_dist = float('inf')
                for t in targets:
                    if t in self.landmarks_dist:
                        d = self.landmarks_dist[t][nr][nc]
                    else:
                        d = abs(nr - t[0]) + abs(nc - t[1])
                    if d < min_dist: min_dist = d
                return min_dist

            possible_moves.sort(key=move_score)
            for m in possible_moves: 
                current_robot_actions.append((m, rid))
            valid_actions.extend(current_robot_actions)
            
        return valid_actions

    def get_transitions(self, state, action_name, robot_id):
        transitions = []
        p_success = self.probs[robot_id]
        
        state_suc, r_suc = self.apply_simulation(state, action_name, robot_id, "success")
        transitions.append((p_success, state_suc, r_suc))
        
        if p_success < 1.0:
            p_fail = 1.0 - p_success
            state_fail, r_fail = self.apply_simulation(state, action_name, robot_id, "fail")
            transitions.append((p_fail, state_fail, r_fail))
        return transitions

    def apply_simulation(self, state, action_type, robot_id, outcome):
        robots, plants, taps, total_need = state
        new_robots = list(robots)
        robot_idx = -1
        for i, r in enumerate(new_robots):
            if r[0] == robot_id: robot_idx = i; break
        rid, (rr, rc), load = new_robots[robot_idx]
        immediate_reward = 0

        if outcome == "success":
            if action_type in ["UP", "DOWN", "LEFT", "RIGHT"]:
                dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[action_type]
                new_robots[robot_idx] = (rid, (rr+dr, rc+dc), load)
            elif action_type == "LOAD":
                new_taps = list(taps)
                for i, (t_pos, t_amt) in enumerate(new_taps):
                    if t_pos == (rr, rc):
                        new_taps[i] = (t_pos, t_amt - 1)
                        new_robots[robot_idx] = (rid, (rr, rc), load + 1)
                        break
                taps = tuple(new_taps)
            elif action_type == "POUR":
                new_plants = list(plants)
                for i, (p_pos, p_need) in enumerate(new_plants):
                    if p_pos == (rr, rc):
                        new_plants[i] = (p_pos, p_need - 1)
                        new_robots[robot_idx] = (rid, (rr, rc), load - 1)
                        total_need -= 1
                        # שימוש ב-R המעודכן!
                        immediate_reward = self.plant_values.get(p_pos, 0)
                        break
                plants = tuple(new_plants)
        else: 
            if action_type == "POUR" and load > 0:
                new_robots[robot_idx] = (rid, (rr, rc), load - 1)
            
        return (tuple(new_robots), plants, taps, total_need), immediate_reward

    def get_safe_dist(self, target, r, c):
        if target in self.landmarks_dist:
            val = self.landmarks_dist[target][r][c]
            if val == float('inf'): return 9999
            return val
        return abs(target[0]-r) + abs(target[1]-c)

    def run_bfs(self, start_node):
        distances = [[float('inf') for _ in range(self.cols)] for _ in range(self.rows)]
        if start_node in self.walls: return distances
        distances[start_node[0]][start_node[1]] = 0
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}
        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    distances[nr][nc] = dist + 1
                    queue.append(((nr, nc), dist + 1))
        return distances