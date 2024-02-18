import gym
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import (TextArea, AnnotationBbox)
import matplotlib.lines as mlines
from .target_finder import TargetFinder


class Trajectory():

  def __init__(self):
      self.type = "both paths"



  def set_type(self, label):
      """
      method alters the class variable "type".
      It is invoked via a RadioButton. The user can select whether he/she wants to see the optimal path,
      the agent's path or both paths.

      :param label: label of RadioButton that specifies which paths the trajectory plot includes.
      """
      self.type = label
      


  def set_background_image(self, plot, max):
    """
    method creates an empty BabyAI grid image. This image is the background of the trajectory plot.

    :param plot: plot in which the background image appears.
    :param max: tuple specifying maximum width and maximum height of grid.
    """
    grid = Grid(max[0], max[1])
    img = grid.render(70)
    plot.imshow(img, extent=[1, max[0]+1, 1, max[1]+1], alpha=0.4)
      


  def set_legend(self, plot, uncertainty_color):
      """
      the method places a legend at the left side of the trajectory plot. 
      The legend contains the symbols that encode the agent's path, the optimal path, 
      the agent's initial and end position, and the target position.

      :param plot: plot to which the legend belongs.
      :param uncertainty_color: color for current step; 
                                color encodes KL divergence between predicted probability distribution and correct probability distribution.
      """
      patch_agent   = patches.Patch(color="tab:blue", label="agent path")
      patch_optimal = patches.Patch(color="yellowgreen", label="optimal path")
      patch_step    = patches.Patch(color=uncertainty_color, label="uncertainty")
      #blue_circle = mlines.Line2D([], [], color="blue", marker="o", markersize=5, label="agent initial position")
      #blue_square = mlines.Line2D([], [], color="blue", marker="s", markersize=5, label="agent end position")
      red_circle     = mlines.Line2D([], [], color="red", marker="o", markersize=5, label="target position")
      blue_triangle  = mlines.Line2D([], [], color="blue", marker=">", markersize=5, label="agent direction")
      green_triangle = mlines.Line2D([], [], color="green", marker=">", markersize=5, label="optimal direction")
      legend_elements = [red_circle, patch_step]
      
      # legend contains only the elements that are currently visible in the plot
      if self.type.__eq__("both paths") or self.type.__eq__("agent path"):
         legend_elements.append(blue_triangle)
         legend_elements.append(patch_agent)
         
      if self.type.__eq__("both paths") or self.type.__eq__("optimal path"):
         legend_elements.append(green_triangle)
         legend_elements.append(patch_optimal)
         
      plot.legend(handles= legend_elements, bbox_to_anchor=(1, 0), loc="lower left", fancybox=True, frameon=True)



  def set_mission(self, plot, mission, maximum):
    """
    the method places the current mission into an Annotationbox. 
    The box is located at the left side of the trajectory plot.

    :param plot: plot to which the legend belongs.
    :param mission: agent's current mission.
    :param max: maximum x and maximum y coordinate of plot.
    """
    factor = 0.25 if maximum[1] == 20 else 0.35
    offsetbox = TextArea("Mission: \n" + mission)
    ab = AnnotationBbox(offsetbox, xy=(maximum[0] / 2.5, maximum[1] + (maximum[1] * factor)), annotation_clip=False)
    plot.add_artist(ab)


  
  def get_grid_size(self, level):
    """
     method determines the grid size; the size varies depending on the level.

    :param level: the name of the BabyAI level, e.g. BabyAI-GoTo-v0.
    :return: a tuple (width, height); can have values (6, 6) or (20, 20).
    """
    level_name = level.split("-")[1]
    
    if level_name in ["GoToObj", "GoToRedBall", "GoToRedBallGrey", "GoToLocal", "PutNextLocal", "PickUpLoc"]:
        size = (6, 6)
    elif level_name in ["GoToObjMaze", "GoTo", "Pickup", "UnblockPickup", "Open", "Unlock", "PutNext", "Synth", 
                        "SynthLoc", "GoToSeq", "SynthSeq", "GoToImpUnlock", "BossLevel"]:
        size = (20, 20)
    else:
        print("Level name does not exist")
        return
    return size


  
  def collect_coordinates(self, offset, env_name, seed, actions):
    """
    the method sets up an environment with a given seed. It determines the mission.
    It redoes the specified actions. After each action, it collects the agent's position. 

    :param offset: offset to get coordinates within a grid field instead of coordinates on the grid fields border.
    :param env_name: BabyAI level name, e.g. BabyAI-GoTo-v0
    :param seed: seed to set up environment.
    :param actions: sequence of actions; after each action, the method stores the agent's coordinates.
    :return: the mission, a list of visited x coordinates, visited y coordinates, and the directions.
    """
    x_coord, y_coord, directions = ([] for _ in range(3))
    
    env = gym.make(env_name)
    gym.Env.seed(env, seed)
    env.seed(seed=seed)
    env.reset()

    # add initial agent position
    x_coord.append(env.agent_pos[0] + offset)
    y_coord.append(env.agent_pos[1] + offset)
    directions.append(env.agent_dir)

    for action in actions:
        _, _, _, _ = env.step(action)
        x_coord.append(env.agent_pos[0] + offset)
        y_coord.append(env.agent_pos[1] + offset)
        directions.append(env.agent_dir)

    return (env.mission, x_coord, y_coord, directions)



  def map_direction_to_marker(self, directions):
    """
    method iterates through a list of numbers. The numbers encode actions
    (0 is right, 1 is down, 2 is left, 3 is up). It adds a direction marker that corresponds to the number to a list.
    Direction marker indicates which direction the agent faces.

    :param directions: list of numbers that encode directions.
    :return: list of direction marker symbols. 
    """
    marker = []
    for dir in directions:
      if dir == 0:
        marker.append(">")
      elif dir == 1:
        marker.append("v")
      elif dir == 2:
        marker.append("<")
      elif dir == 3:
        marker.append("^")
      else:
        print("number not in BabyAI direction numbers")
    return marker



  def place_direction_marker(self, plot, x_coord, y_coord, step_idx, directions, label, color):
    """
    the method visualizes the agent's direction. It places a direction marker in the grid.
    The direction marker indicates which direction the agent faces.

    :param plot: plot where to place the marker.
    :param x_coord: list of x coordinates that belong to a trajectory/ path. 
    :param y_coord: list of y coordinates that belong to a trajectory/ path.  
    :param step_idx: index of step in list of steps.
    :param directions: sequence of directions obtained while tracing a path.
    :param label: label for the direction marker.
    :param color: direction marker's color.
    """
    marker = self.map_direction_to_marker(directions)
    plot.scatter(x_coord[step_idx], y_coord[step_idx], label=label, c=color, marker=marker[step_idx], zorder=5.0, s=80)



  def place_arrow(self, plot, x, y, direction, agent_orientation, color, offset):
    """
    method places an arrow in a grid. The arrow indicates into which direction the agents turn.

    :param plot: a plot where to place the data.
    :param x: the agent's current x coordinate.
    :param y: the agent's current y coordinate.
    :param direction: the direction into which the agent turns.
    :param agent_orientation: the direction that the agent currently faces.
    :param color: the color for current step; color encodes KL divergence.
    :param offset: offset coordinates used for placing the arrow properly. 
    """
    width = 1.9
    x1 = x - offset
    y1 = y - offset
    x2 = x + (1 - offset)
    y2 = y + (1 - offset) 

    # turn left
    if direction == 0:
      if agent_orientation == 0:
        plot.annotate("", xy=(x2, y1), xytext=(x1, y2), 
                      arrowprops=dict(arrowstyle="->", color=color, lw= width, connectionstyle="angle3, angleA=0, angleB=90"))

      elif agent_orientation == 1:
        plot.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                      arrowprops=dict(arrowstyle="->", color=color, lw=width, connectionstyle="angle3, angleA=90, angleB=0"))

      elif agent_orientation == 2:
          plot.annotate("", xy=(x1, y2), xytext=(x2, y1), 
                        arrowprops=dict(arrowstyle="->", color=color, lw=width, connectionstyle="angle3, angleA=0, angleB=90"))

      elif agent_orientation == 3:
          plot.annotate("", xy=(x1, y1), xytext=(x2, y2), 
                        arrowprops=dict(arrowstyle="->", color=color, lw=width, connectionstyle="angle3, angleA=90, angleB=0"))
      
    # turn right
    if direction == 1:
      if agent_orientation == 0 or agent_orientation == 3:
          plot.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                        arrowprops=dict(arrowstyle="->", color=color,  lw=width, connectionstyle="angle3, angleA=0, angleB=90"))

      elif agent_orientation == 1:
          plot.annotate("", xy=(x1, y2), xytext=(x2, y1), 
                        arrowprops=dict(arrowstyle="->", color=color,  lw=width, connectionstyle="angle3, angleA=90, angleB=0"))

      elif agent_orientation == 2:
          plot.annotate("", xy=(x1, y1), xytext=(x2, y2), 
                        arrowprops=dict(arrowstyle="->", color=color,  lw=width, connectionstyle="angle3, angleA=0, angleB=90"))



  def draw_line_collection(self, plot, x_coord, y_coord, step_idx, path_color, step_color):
    """
     the method creates a line collection that shows a trajectory/ path. Each single line of the collection
     represents a step of the path. It enables to use different colors for different steps.

    :param plot: plot where to place the line collection.
    :param x_coord: list of x coordinates that belong to a trajectory/ path. 
    :param y_coord: list of y coordinates that belong to a trajectory/ path.  
    :param step_idx: index of step in list of steps.
    :param path_color: color for line that encodes the predicted or the correct path in a grid.
    :param step_color: color for current step; 
                       color encodes KL divergence between predicted probability distribution and correct probability distribution.
    """
      
    # draw path
    points = np.array([x_coord, y_coord]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # color current step in another color; 
    # in the optimal path, this color encodes the uncertainty value
    colors = [path_color for _ in range(len(segments))]
    colors[step_idx] = step_color 

    lc = LineCollection(segments, lw=3.5, color=colors)
    lc.set_array(y_coord)
    plot.add_collection(lc)



  def visualize_trajectory(self, env_name, plot, actions, actions_true, seed, glyph_no, 
                           step_wise=False, step_idx=None, color=None):
    """
    method visualizes the agent's path and the optimal path. 

    :param env_name: BabyAI level name.
    :param plot: an empty plot where to place the data.
    :param actions: action sequence that the agent tries until it reaches the maximum number of steps and fails.
    :param actions_true: true actions specified in demo.
    :param seed: seed of an episode.
    :param glyph_no: identifier.
    :param step_wise: if true, the method creates a seperate line for each step.
                      if false, single line connects all trajectory coordinates.
    :param step_idx: index of step in list of steps.
    :param color: color for current step; 
                  color encodes KL divergence between predicted probability distribution and correct probability distribution.
    :return: a plot visualizing the grid and the agent's path.
    """
  
    plot.clear()
    max = self.get_grid_size(env_name)
    plot.set_xlim(1, max[0] + 1, auto=False)
    plot.set_ylim(max[1] + 1, 1, auto=False)
    
    # create background image
    self.set_background_image(plot, max)
    
 
    # offset is used to plot the dots and lines within a grid field
    # otherwise the lines and dots are plotted on grid lines
    # new offset is used to avoid overplotting if agent's trajectory and optimal trajectory are identical
    offset, offset_true = 0.4, 0.6
    #print(actions)
    (mission, x_coord, y_coord, directions) = self.collect_coordinates(offset, env_name, seed, actions)
    (_, x_coord_true, y_coord_true, directions_true) = self.collect_coordinates(offset_true, env_name, seed, actions_true)
    
    
    # plot target position 
    TargetFinder().determine_target_position(env_name, seed)
    #plot.scatter(target_pos[0] + offset, target_pos[1] + offset, label="target position", c="red", marker="o")  
    plot.scatter(x_coord_true[-1] +0.1, y_coord_true[-1] +0.1, label="target position", c="red", marker="o", zorder=5.0)  

    if step_wise and (step_idx != None) and (color != None):
        self.visualize_as_plot(plot, glyph_no, actions, actions_true, x_coord, y_coord, x_coord_true, y_coord_true, step_idx,
                               directions, directions_true, max, mission, color)
    else: 
        self.visualize_as_glyph(plot, glyph_no, x_coord, y_coord, x_coord_true, y_coord_true)
    
    return plot



  def visualize_as_plot(self, plot, step_no, actions, actions_true, x_coord, y_coord, x_coord_true, y_coord_true, step_idx,
                        directions, directions_true, max, mission, color):
     """
     method visualizes the trajectory in a stand-alone plot.  

     :param plot: a plot where to place the data.
     :param step_no: identifier.
     :param x_coord: list of x coordinates that belong to a predicted trajectory/ path. 
     :param y_coord: list of y coordinates that belong to a predicted trajectory/ path. 
     :param x_coord_true: list of x coordinates that belong to the correct trajectory/ path specified in training data. 
     :param y_coord_true: list of y coordinates that belong to correct trajectory/ path specified in training data.  
     :param step_idx: index of step in list of steps.
     :param directions: sequence of directions obtained while tracing a predicted path.
     :param directions_true: sequence of directions obtained while tracing the correct path.
     :param max: maximum x and maximum y coordinate of plot.
     :param mission: agent's current mission.
     :param color: color for current step; 
                  color encodes KL divergence between predicted probability distribution and correct probability distribution.
     """
     plot.set_title("Step " + str(step_no), loc="left")

     if self.type.__eq__("both paths") or self.type.__eq__("agent path"):
        self.draw_line_collection(plot, x_coord, y_coord, step_idx, "tab:blue", "tab:blue")
        self.place_direction_marker(plot, x_coord, y_coord, step_idx, directions, "direction", "blue")

        # place arrow if agent turns left (0) or right (1)
        if actions[step_idx] in [0, 1]:
            self.place_arrow(plot, x_coord[step_idx], y_coord[step_idx], actions[step_idx], directions[step_idx], "tab:blue", 0.4)
        
     if self.type.__eq__("both paths") or self.type.__eq__("optimal path"):
         self.draw_line_collection(plot, x_coord_true, y_coord_true, step_idx, "yellowgreen", color)
         self.place_direction_marker(plot, x_coord_true, y_coord_true, step_idx, directions_true, "direction true", "green")

         # place arrow if agent turns left (0) or right (1)
         if actions_true[step_idx] in [0, 1]:
            self.place_arrow(plot, x_coord_true[step_idx], y_coord_true[step_idx], actions_true[step_idx], directions_true[step_idx], color, 0.6)
          
     self.set_legend(plot, color)
     self.set_mission(plot, mission, max)

  

  def visualize_as_glyph(self, plot, glyph_no, x_coord, y_coord, x_coord_true, y_coord_true):
      """
      this method creates a trajectory plot. This plot is shown as a glyph.

      :param plot: a plot where to place the data.
      :param glyph_no: identifier.
      :param x_coord: list of x coordinates that belong to a predicted trajectory/ path. 
      :param y_coord: list of y coordinates that belong to a predicted trajectory/ path. 
      :param x_coord_true: list of x coordinates that belong to the correct trajectory/ path specified in training data. 
      :param y_coord_true: list of y coordinates that belong to correct trajectory/ path specified in training data.  
      """
      plot.set_title("Episode: " + str(glyph_no), fontsize=14)

      # plot agent's start position
      plot.scatter(x_coord[0], y_coord[0], label="agent initial position", c="blue", marker="o", zorder=5.0)

      # plot agent's final position
      plot.scatter(x_coord[-1], y_coord[-1], label="agent end position", c="blue", marker="s", zorder=5.0)

      # plot agent's path
      plot.plot(x_coord, y_coord, lw = 8.0)

      # plot optimal path
      plot.plot(x_coord_true, y_coord_true, lw = 8.0, c="fuchsia")



 

    

 
    

    

    

    

   
