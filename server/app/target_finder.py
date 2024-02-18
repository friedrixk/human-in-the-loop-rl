import gym


class TargetFinder():
    
 def limit_search_span(self, location, agent_pos, agent_dir):
     """
     method determines the rectangle where the possible target can be.
     Some missions define locations like "on your left" or "behind you".
     So, the search space is not the whole grid anymore.

     :param location: string that defines a direction ("left", "right", "behind", "front").
     :param agent_pos: agent's position.
     :param agent_dir: direction the agent currently faces.
     :return: list [(xmin, xmax), (ymin, ymax)] that contains 
              four coordinates that define the rectangular search space.
     """
     if location.__eq__("left"):
         if agent_dir == 0:
             search_span = [(0, 20), (0, agent_pos[1])]        
         elif agent_dir == 1:
             search_span = [(agent_pos[0], 20), (0, 20)]
         elif agent_dir == 2:
             search_span = [(0, 20), (agent_pos[1], 20)]
         elif agent_dir == 3:
             search_span = [(0, agent_pos[0]), (0, 20)]

     if location.__eq__("right"):
         if agent_dir == 0:
             search_span = [(0, 20), (agent_pos[1], 20)]        
         elif agent_dir == 1:
             search_span = [(0, agent_pos[0]), (0, 20)]
         elif agent_dir == 2:
             search_span = [(0, 20), (0, agent_pos[1])] 
         elif agent_dir == 3:
             search_span = [(agent_pos[0], 20), (0, 20)]

     if location.__eq__("behind"):
         if agent_dir == 0:
             search_span = [(0, agent_pos[0]), (0, 20)]        
         elif agent_dir == 1:
             search_span = [(0, 20), (0, agent_pos[1])] 
         elif agent_dir == 2:
             search_span = [(agent_pos[0], 20), (0, 20)]
         elif agent_dir == 3:
             search_span = [(0, 20), (agent_pos[1], 20)] 

     if location.__eq__("front"):
         if agent_dir == 0:
             search_span = [(agent_pos[0], 20), (0, 20)]        
         elif agent_dir == 1:
             search_span = [(0, 20), (agent_pos[1], 20)] 
         elif agent_dir == 2:
             search_span = [(0, agent_pos[0]), (0, 20)] 
         elif agent_dir == 3:
             search_span = [(0, 20), (0, agent_pos[1])] 

     return search_span



 def loc_spec_mission(self, mission, agent_pos, agent_dir):
    """
    method filters a mission that contains a location specifier. It invokes limit_search_span() method.

    :param mission: the agent's mission.
    :param agent_pos: agent's position.
    :param agent_dir: direction the agent currently faces.
    :return: tuple that contains the mission without location specifier and a 
             list [(xmin, xmax), (ymin, ymax)] that contains four coordinates 
             that define the rectangular search space.
    """
    location = list(filter(lambda x: x in ["left", "right", "front", "behind"], mission) )[0]
    mission  = list(filter(lambda x: x not in ["on", "your", "left", "right", "in", "front", "of", "you", "behind"], mission) )
    search_span = self.limit_search_span(location, agent_pos, agent_dir)

    return (mission, search_span)



 def find_target(self, mission, agent_pos, agent_dir):
    """
    method to split the mission sentence. It looks for the words that describe the agent's target object.

    :param mission: the string encoding the agent's mission.
    :param agent_pos: agent's position.
    :param agent_dir: direction the agent currently faces.
    :return: tuple with color, type of the target object, and search span. 
    """
    search_span = [(0, 20), (0, 20)]

    if "after you" in mission:
        mission = mission.split("after you")[0]
    elif "then" in mission:
        mission = mission.split("then")[1]
    if "and" in mission:
        mission = mission.split("and")[-1]
    
    
    mission = list(filter(lambda x: x not in ["a", "the", "go", "to", "pick", "up", "open"], mission.split(" ")) )

    if "put" in mission:
        mission = mission[mission.index("next") + 1 : ]
        
    if "left" in mission or "right" in mission or  "front" in mission or  "behind" in mission:
        print(mission)
        (mission, search_span) = self.loc_spec_mission(mission, agent_pos, agent_dir)
        print(mission)

    color = list(filter(lambda x: x in ["red", "green", "blue", "purple", "yellow", "grey"], mission))
    color = color[0] if len(color) > 0 else None
 
    obj_type = list(filter(lambda x: x in ["door", "ball", "box", "key"], mission))[0]
    #print(mission, "target: ", color , obj_type)
    return (color, obj_type, search_span)

        

 def find_object_pos(self, env, color, target_type, search_span):
    """
    method to find all grid positions at which a target object is placed.

    :param env: level name.
    :param color: target color.
    :param target_type: type of target, e.g., ball, box,...
    :param search_span: list that contains four coordinates that define the rectangular search space.
    :return: all target positions.
    """
    [(xmin, xmax), (ymin, ymax)] = search_span
    
    positions = []
    for j in range(env.grid.height):                                         
        for i in range(env.grid.width):
          if j > ymin and j <= ymax and i > xmin and i <= xmax:
            o = env.grid.get(i, j)
            if o != None:
             if o.type == target_type:
                if color == None:
                    positions.append([i,j])
                    #print(o.type)
                elif o.color == color:
                     positions.append([i,j])
                     #print(o.type, o.color)
    #print("pos: ", positions)
    return positions



 def manhattan_distance(self, p1, p2):
    """
     method calculates manhattan distance between two points.

     :param p1: first point.
     :param p2: second point.
     :return: manhattan distance between the p1 and p2.
    """
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])



 def find_nearest_target(self, agent_pos, possible_targets):
    """
    method to determine which target object has the smallest distance to the agent.

    :param agent_pos: agent's current position.
    :param possible_targets: coordinates at which a target is placed.
    :return: coordinates of the target with the smallest distance to the agent.
    """
    # if there is only one possible target, we don't need to select one out of all possible targets
    if len(possible_targets) == 1:
        #print("target coordinates: ", possible_targets[0])
        return possible_targets[0]
    
    min_dist = 100
    target = []
    for coord in possible_targets:
       if self.manhattan_distance(agent_pos, coord) < min_dist:
           min_dist = self.manhattan_distance(agent_pos, coord)
           target = coord
    #print("target coordinates: ", target)
    return target



 


 def determine_target_position(self, env, seed):
    """
    method to find the coordinates of the agent's target.

    :param env: level name.
    :param seed: seed to set up a specific episode.
    :return: coordinates of the target with the smallest distance to the agent.
    """
    # some missions for testing
    #m1 = "go to the red ball"
    #m2 = "open the door on your left"
    #m3 = "put a ball next to the blue door"
    #m4 = "open the yellow door and go to the key behind you"
    #m5 = "put a ball next to a purple door after you put a blue box next to a grey box and pick up the purple box"
    #m6 = "go to the yellow key on your left"
    #m7 = "go to the red ball on your left"
  
      
    env = gym.make(env)
    gym.Env.seed(env, seed)
    env.seed(seed)
    env.reset()
    mission = env.mission
    #print("mission: ", mission, type(mission))

    (color, obj_type, search_span) = self.find_target(mission, env.agent_pos, env.agent_dir)
    possible_targets = self.find_object_pos(env, color , obj_type, search_span)
    final_target = self.find_nearest_target(env.agent_pos, possible_targets)
    #print("final target: ", final_target)
    return final_target



# for testing 

#parser = ArgumentParser()
#parser.add_argument("--file_path", help="specify path to file that stores the visualization data")
#parser.add_argument("--save_path", help="specify path for saving the plots")
#parser.add_argument("--env", required=True,
 #                   help="name of the environment to be run (REQUIRED)")
#parser.add_argument(
    #"--seed",
    #type=int,
    #help="random seed to generate the environment with",
    #default=-1
#)
#args = parser.parse_args()

#target_finder = TargetFinder()
#print(target_finder.determine_target_position(args.env, args.seed))