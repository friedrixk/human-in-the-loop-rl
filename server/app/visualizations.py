from argparse import ArgumentParser
import copy
import matplotlib
from matplotlib.container import BarContainer
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from app.sampler import Sampler
from app.color import Color
from app.trajectory import Trajectory
from app.file_handler import FileHandler
from app.target_finder import TargetFinder
from app.utility_vis import UtilityVis
from IPython import display



parser = ArgumentParser()
parser.add_argument("--env", required=True, help="name of the environment to be run (REQUIRED)")
parser.add_argument("--file_path", help="specify path to file that stores the visualization data")
parser.add_argument("--save_path", help="specify path for saving the plots")



class PlotGrid():

 def __init__(self):


    #matplotlib.style.use("Solarize_Light2")
    #matplotlib.style.use("ggplot")
    #matplotlib.style.use("dark_background")
    matplotlib.style.use("seaborn-dark")
    self.fig = plt.figure(figsize=(9.5, 9.5), dpi=120)
   
    # grid contains 5 plots
    self.plot1 = plt.subplot2grid((100,100), (0,0), rowspan=26, colspan=42)
    self.plot2 = plt.subplot2grid((100,100), (0,57), rowspan=26, colspan=42)
    self.plot3 = plt.subplot2grid((100,100), (38,0), rowspan=26, colspan=42)
    self.plot4 = plt.subplot2grid((100,100), (38,57), rowspan=26, colspan=42)
    self.plot5 = plt.subplot2grid((100,100), (74,0), rowspan=26, colspan=42)
    self.plot6 = plt.subplot2grid((100,100), (74,60), rowspan=26, colspan=26)

    # workaround matplotlib/ipympl #290
    if UtilityVis().type_of_script() == 'jupyter':
        display.display(self.fig.canvas)
        self.fig.canvas._handle_message(self.fig.canvas, {'type': 'send_image_mode'}, [])
        self.fig.canvas._handle_message(self.fig.canvas, {'type':'refresh'}, [])
        self.fig.canvas._handle_message(self.fig.canvas,{'type': 'initialized'},[])
        self.fig.canvas._handle_message(self.fig.canvas,{'type': 'draw'},[])



 def set_plotdata(self, env_name, fail_seeds, returns, action_distribution): 
     """
     method takes as input parameters the data to visualize and initializes the class variables.

     :param env_name: name of agent's environment, e.g. BabyAI-GoToObj-v0.
     :param fail_seeds: the seeds of failed episodes.
     :param returns: the return values that the agent achieves in an episode.
     :param action_distribution: the failed action sequences tried by the agent during an episode.
     """

     self.env = env_name
     self.fail_seeds = fail_seeds
     self.returns = returns
     self.action_distribution = action_distribution
     


 def set_env(self, env_name):
     """
     setter for class variable env.

     :param env_name: BabyAI level name.
     """
     self.env = env_name



 def preprocessing(self, fail_actions):
    list_preprocessed = [[fail_actions[j][i] for j in range(len(fail_actions))] for i in range(len(fail_actions[0]))]
    #print("preprocessing done: ", list_preprocessed)
    return list_preprocessed



 #to do 
 def calculate_manhattan_distance(self, path, target):
    """
     method calculates the manhattan distance from every point in a list to the target.

     :param path: list of grid points. 
     :param target: the coordinates of the agent's target.
     :return: list containing all distances from each point to the target.
    """
    dist = []
    for p in path:
        new_dist = abs(p[0]-target[0]) + abs(p[1]-target[1])
        dist.append(new_dist)
    
    return dist


 def accumulate_frequency(self, action_numbers):
    """
     method counts how often an action appears in a list of actions.

     :param action_numbers: a list containing numbers; each number encodes an action.
     :return: the list containing the frequency of each action.
    """
    left = right = forward = toggle = pickup = drop = 0
    action_frequency = []

    for i in action_numbers:
        if(i == 0):
            left += 1
        elif(i == 1):
            right += 1
        elif(i == 2):
            forward += 1
        elif(i == 3):
            pickup += 1
        elif(i == 4):
            drop += 1
        elif(i == 5):
            toggle += 1
        else:
            return "action number is not part of environment's action enumeration"
        
    action_frequency = [left, right, forward, pickup, drop, toggle]
    #print("frequency of actions: ", action_frequency)
    return action_frequency


 def get_label(self, action_number):
    """
    method to map a number to the corresponding action.

    :param action_number: integer that encodes an action.
    :return: the name of the action (e.g., "left",  "right", ...).
    """
    if(action_number == 0):
        label = "left"
    elif(action_number == 1):
        label = "right" 
    elif(action_number == 2):
         label = "forward"
    elif(action_number == 3):
         label = "pickup" 
    elif(action_number == 4):
         label = "drop" 
    elif(action_number == 5):
         label = "toggle"
    else:
        return "action number is not part of environment's action enumeration"
    return label


    

 def retrace_agent_path(self, env, seed, actions):
    """
    method to retrace the agent's path in order to collect the visited coordinates/ positions.

    :param env: name of the environment/ level.
    :param seed: seed of the episode for which we want to know the agent's coordinates.
    :param actions: the agent's actions that it chooses during an episode. 
    :return: coordinates of agent's path
    """
    env = gym.make(env)
    gym.Env.seed(env, seed)
    env.seed(seed)
    env.reset()

    path = []
    path.append(env.agent_pos)
       
    for action in actions:
        _, _, _, _ = env.step(action)
        path.append(env.agent_pos)
    
    #print("seed: ", seed, "\npath: ", path)
    return path



 

 # plot 1
 def create_success_rate_pie(self, plot, returns):
    """
    method calculates the success rate and fail rate; it shows the data in a pie chart.

    :param plot: an empty plot where to place the data.
    :param returns: return values of successful and failed episodes.
    :return: a plot containing the pie chart.
    """
    
    #print(len(returns))
    success_rate = np.mean([1 if r > 0 else 0 for r in returns]) *100 
    fail_rate = 100 - success_rate
    percentages = [success_rate, fail_rate]
  
    #print(success_rate)

    plot.set_title("Success Rate\n", fontweight="bold")
    labels = ["success", "failure"]
    colors = ["yellowgreen", "crimson"]
    explode = (0, 0)
    text_style = {"fontsize": 10, "color":"black"}

    plot.pie(percentages, explode=explode, labels=labels, colors=colors,
     autopct="%.0f%%", 
     pctdistance=1.15,
     shadow=False,
     textprops=text_style,
     labeldistance=1.38,
     wedgeprops={"linewidth": 3.0, "edgecolor": "white"},)

    centre_circle = plt.Circle((0,0), 0.67, color="grey", fc="white", linewidth=0.4)
    plot.add_artist(centre_circle)
    plot.axis("equal")

    return plot


 # plot 2
 def create_return_plot(self, plot, returns):
    """
    method creates a bar chart; each bar represents a return value.
    It calculates the mean of the return values and visualizes the mean value as a horizontal line.

    :param plot: an empty plot where to place the data.
    :param returns: return values of successful and failed episodes.
    :return: a plot containing the bar chart.
    """
    plot.clear()
    plot.set_title("Returns per Episode", fontweight="bold")
    plot.set_xlabel("episodes", loc="right")
    plot.set_ylabel("return")
    plot.grid(axis="y")

    x_range = np.arange(len(returns))
    bar_width = 0.5

    #print("x range: ", x_range)
    return_mean = [np.mean(returns)] * len(returns)

    plot.bar(x_range, returns, bar_width, color="tab:blue")

    # plot line that indicates the mean value of the return values/ height of the bars
    x_range_line = np.append(x_range, len(returns)+1)
    y_range_line = np.append(return_mean, return_mean[0])
    plot.plot(x_range_line, y_range_line, color="tab:blue")
    plot.annotate("mean", xy=(x_range_line[-1] , y_range_line[0]), xytext= (x_range_line[-1] + 0.2, y_range_line[0]),  
                  arrowprops=dict(facecolor='black', arrowstyle="->") )
    
    # tooltip
    c = mplcursors.cursor(plot, hover=mplcursors.HoverMode.Transient)
    @c.connect("add")
    def _(sel):
      if type(sel.artist) == BarContainer:
        x, _, width, _ = sel.artist[sel.index].get_bbox().bounds
        sel.annotation.set(position= (x + width / 2, 0))
        sel.annotation.get_bbox_patch().set(height=1.0, width=1.0, boxstyle="square", fc="white", alpha=1.0)
        sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=1.0)
      
        bar = sel.artist[sel.target.index]
        sel.annotation.set(text= "\n return:  " + str(round(returns[round(bar.get_x())], 3)) +" \n" )

      else:
          sel.annotation.set_visible(False)
      
    return plot


 # alternative to plot 3 
 def create_stacked_barchart(self, plot, action_distribution, x_labels):
    """
    method that creates an alternative to the stacked horizontal bar chart.

    :param plot: an empty plot where to place the data.
    :param action_distribution: action sequences that the agent tries until it reaches the maximum number of steps and fails.
    :param x_labels: data represented on x axis.
    :return: a plot containing the bar chart.
    """
    plot.set_title("Actions per Episode", fontweight="bold")
    plot.set_xlabel("actions")
    plot.set_ylabel("steps")

    width, height, offset = 0.15, 1, 0
    list_preprocessed = self.preprocessing(action_distribution)
 
    for data in list_preprocessed:
        color_list = [Color().map_action_to_color(data[j]) for j in range(len(data))]
        #print(color_list)
        plot.bar(x_labels, height, width, bottom=offset, color=color_list)
        offset += height

    return plot




 # plot 3
 def create_stacked_horizontal_barchart(self, plot, trajectory_plot, seeds, action_distribution,
                                        trigger_trajectory_update=True, legend_loc=(1,0), ncol=1):
    """
    method that creates a visualization for the fail actions. 
    It aranges the bars vertically. Each bar shows an action sequence.
    If the user hovers over a bar, the method initiates the creation of the trajectory plot.

    :param plot: an empty plot where to place the data.
    :param trajectory_plot: visualization that is linked with the tooltip of the stacked bar chart.
    :param seeds: seeds of the failed actions.
    :param action_distribution: action sequences that the agent tries until it reaches the maximum number of steps and fails.
    :return: a plot containing the bar chart.
    """
    plot.clear()
    plot.set_title("Actions per Episode", fontweight="bold")
    plot.set_xlabel("steps", loc="right")
    plot.set_ylabel("episodes")

    y_labels = np.arange(len(seeds))
    height, offset = 0.8, 0.0

    # creating lists of same length to get bars of same length
    # if there is no value at a step because the mission finished earlier, color grey is displayed
    action_distribution_copy = copy.deepcopy(action_distribution)
    max_length = max([len(list) for list in action_distribution_copy])
    for list in action_distribution_copy:
        while len(list) < max_length:
          list.append(None)


    list_preprocessed = self.preprocessing(action_distribution_copy)
    #print("list_preprocessed: ", list_preprocessed)

        
    for data in list_preprocessed:
        color_list = [Color().map_action_to_color(data[j]) for j in range(len(data))]
        #print(color_list)
        plot.barh(y_labels, 1, height, left=offset, color=color_list, edgecolor="white")
        offset += height 

    # legend
    colors = ["tab:orange", "tab:cyan", "tab:blue", "tab:olive", "tab:purple", "tab:gray"]
    actions = ["left", "right", "forward", "pickup", "drop", "toggle"]
    patch = []
    for i, c in enumerate(colors):
     new_patch = mpatches.Patch(color=c, label=actions[i])
     patch.append(new_patch)
    plot.legend(handles=patch, bbox_to_anchor=legend_loc, ncol=ncol, loc="lower left", fancybox=True, frameon=True)

    # tooltip
    if trigger_trajectory_update:
     c = mplcursors.cursor(plot, hover=True)
     @c.connect("add")
     def _(sel):
      x, _, width, _ = sel.artist[sel.index].get_bbox().bounds
      sel.annotation.set(position= (x + width / 2, -3))
      sel.annotation.get_bbox_patch().set(height=2.5, width=2.0, boxstyle="square", fc="white", alpha=1.0)
      sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=1.0)

      if type(sel.artist) == BarContainer:
        bar = sel.artist[sel.target.index]
        sel.annotation.set(text="\n seed:  "+ str(seeds[round(bar.get_y())]) 
                              + " \n replay_no: " + str(self.fail_seeds.index(seeds[round(bar.get_y())])))
        #trajectory_plot.clear()
        self.create_trajectory_plot(trajectory_plot, action_distribution[round(bar.get_y())], seeds[round(bar.get_y())])
    
    return plot


 # plot 4
 def create_grouped_barchart(self, plot, seeds, action_distribution):
    """
    method that creates a grouped bar chart for the distribution of fail actions.

    :param plot: an empty plot where to place the data.
    :param seeds: seeds of the failed actions.
    :param action_distribution: action sequences that the agent tries until it reaches the maximum number of steps and fails.
    :return: a plot containing the bar chart.
    """
    plot.set_title("Fail Actions - Distribution", fontweight="bold")
    plot.set_xlabel("episodes", loc="right")
    plot.set_ylabel("incidence")
    plot.grid(axis="y")

    width = 0.2

    for i in np.arange(len(seeds)):
      action_frequency = self.accumulate_frequency(action_distribution[i])
      left = np.array(action_frequency[0])
      right = np.array(action_frequency[1])
      forward = np.array(action_frequency[2])
      pickup = np.array(action_frequency[3])
      drop = np.array(action_frequency[4])
      toggle = np.array(action_frequency[5])
       
      plot.bar(i - (2*width), left, width, label="left", color="tab:orange")
      plot.bar(i - width, right, width, label="right", color="tab:cyan")
      plot.bar(i , forward, width, label="forward", color="tab:blue")
      plot.bar(i + width, pickup, width, label="pickup", color="tab:olive")
      plot.bar(i + (2*width), drop, width, label="drop", color="tab:purple")
      plot.bar(i + (3*width), toggle, width, label="toggle", color="tab:gray")
      
      
    # legend
    colors = ["tab:orange", "tab:cyan", "tab:blue", "tab:olive", "tab:purple", "tab:gray"]
    actions = ["left", "right", "forward", "pickup", "drop", "toggle"]
    patch = []
    for i, c in enumerate(colors):
     new_patch = mpatches.Patch(color=c, label=actions[i])
     patch.append(new_patch)
    plot.legend(handles=patch, bbox_to_anchor=(1, 0), loc="lower left", fancybox=True, frameon=True)
    

    # tooltip
    c = mplcursors.cursor(plot, hover=mplcursors.HoverMode.Transient)
    @c.connect("add")
    def _(sel):
      x, _, width, _ = sel.artist[sel.index].get_bbox().bounds
      sel.annotation.set(position= (x + width / 2, -3))
      sel.annotation.get_bbox_patch().set(height=2.5, width=2.0, boxstyle="square", fc="white", alpha=1.0)
      sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=1.0)
      
      # tooltip appears only for cyan, yellow, and orange bars to ensure the correct seed and replay number 
      # otherwise the x position of the bar does not correspond to the correct seed and replay number 
      allowed_rgba_values = [(0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0,  0.6470588235294118, 0.0, 1.0)]
      bar_color = sel.artist[sel.target.index].get_facecolor()
      
      if type(sel.artist) == BarContainer and bar_color in allowed_rgba_values:
        bar = sel.artist[sel.target.index]
        sel.annotation.set(text=" \n seed: "+ str(seeds[round(bar.get_x())]) 
                               + " \n replay_no: " + str(self.fail_seeds.index(seeds[round(bar.get_x())]) ))
      else:
        sel.annotation.set_visible(False)

    return plot


 


 # plot 5
 def create_distance_plot(self, plot, trajectory_plot, seeds, action_distribution, 
                          trigger_trajectory_update=True, legend_loc=(1,0), color_blind=False):
    """
    method creates a plot that visualizes whether a step reduced the distance to the target or increased the distance.
    If the user hovers over a bar, the method initiates the creation of the trajectory plot.

    :param plot: an empty plot where to place the data.
    :param trajectory_plot: visualization that is linked with the tooltip of the distance plot.
    :param seeds: seeds of the failed actions.
    :param action_distribution: action sequences that the agent tries until it reaches the maximum number of steps and fails.
    :return: a plot containing the bar chart.

    """
    plot.clear()
    plot.set_title("Distance to Target", fontweight="bold")
    plot.set_xlabel("steps", loc="right")
    plot.get_xaxis().set_visible(False)
    plot.set_ylabel("episodes")
    
    target_finder = TargetFinder()
    target_pos, paths = [], [] 
    
    color_scheme = ["yellowgreen", "crimson", "tab:blue", "tab:orange"]
    color_encoding = [color_scheme[0] if not color_blind else color_scheme[2], 
                      color_scheme[1] if not color_blind else color_scheme[3]]

    for i, list in enumerate(action_distribution):
        paths.append(self.retrace_agent_path(self.env, seeds[i], list))
        target_pos.append(target_finder.determine_target_position(self.env, seeds[i]))

    y_labels = np.arange(len(seeds))
    height, offset = 0.8, 0
    
    dist = [self.calculate_manhattan_distance(p, target_pos[i]) for i, p in enumerate(paths)]
    #print("dist: ", dist)
    color_list = [[color_encoding[0] if dist[j][i] > dist[j][i+1] else color_encoding[1] for i in range(len(dist[j])-1)] for j in range(len(dist))]   
    #print("color_list: ", color_list)
    
    # creating lists of same length to get bars of same length
    # if there is no value at a step because the mission finished earlier, color grey is displayed
    max_length = max([len(list) for list in color_list])
    for list in color_list:
        while len(list) < max_length:
          list.append("lightgrey")
    
    # reorder stepwise: n-th steps are in the same list
    color_list_reordered = [[color_list[i][j] for i in range(len(color_list))] for j in range(len(color_list[0]))]  
    #print("color_list_reordered: ", color_list_reordered)
    
    for colors in color_list_reordered:
      plot.barh(y_labels, 1, height, left=offset, color=colors, edgecolor="white")
      offset += height
    
    # legend
    patch = []
    patch_green = mpatches.Patch(color=color_encoding[0], label="distance decreased")
    patch.append(patch_green)
    patch_red = mpatches.Patch(color=color_encoding[1], label="distance unchanged\n or increased")
    patch.append(patch_red)
    plot.legend(handles=patch, bbox_to_anchor=legend_loc, loc="lower left", fancybox=True, frameon=True)

    # tooltip
    if trigger_trajectory_update:
     c = mplcursors.cursor(plot, hover=True)
     @c.connect("add")
     def _(sel):
      sel.annotation.set(position= (20, -1.5))
      sel.annotation.get_bbox_patch().set(height=2.5, width=2.0, boxstyle="square", fc="white", alpha=1.0)
      sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=1.0)
      
      if type(sel.artist) == BarContainer:
        bar = sel.artist[sel.target.index]
        distance = dist[round( bar.get_y())][-1]
        sel.annotation.set(text= "\n seed:  "+ str(seeds[round(bar.get_y())]) 
                                 + " \n replay_no: " + str(self.fail_seeds.index(seeds[round(bar.get_y())]))
                                 + "\n remaining distance to target\n at the end:  "+ str(distance) + " \n")
        
        self.create_trajectory_plot(trajectory_plot, action_distribution[round(bar.get_y())], seeds[round(bar.get_y())])

    return plot
        



  # plot 6 
 def create_trajectory_plot(self, plot, fail_actions, seed):
    """
    method to visualize the agent's path. 

    :param plot: an empty plot where to place the data.
    :param fail_actions: action sequence that the agent tries until it reaches the maximum number of steps and fails.
    :param seed: seed of the failed episode.
    :return: a plot visualizing the grid and the agent's path.
    """
   
    plot.clear()
    plot.set_title("Trajectory", fontweight="bold")
    max = Trajectory().get_grid_size(self.env)
    plot.set_xlim(1, max[0] + 1, auto=False)
    plot.set_ylim(max[1] + 1, 1, auto=False)
    plot.grid()

    # offset is used to plot the dots and lines within a grid field, 
    # otherwise the lines and dots are plotted on grid lines
    offset = 0.5
    x_coord, y_coord = [], []
    print("start: determine trajectory")
    
    env = gym.make(self.env)
    gym.Env.seed(env, seed)
    env.seed(seed)
    env.reset()
    
    x_coord.append(env.agent_pos[0] + offset)
    y_coord.append(env.agent_pos[1] + offset)
    
    for action in fail_actions:
        _, _, _, _ = env.step(action)
        x_coord.append(env.agent_pos[0] + offset)
        y_coord.append(env.agent_pos[1] + offset)
        
    print("finished")  
    #print(x_coord)   
    #print(y_coord)
    
    # plot agent's start position
    plot.scatter(x_coord[0], y_coord[0], label= "agent initial position", c="blue", marker="o")

    # plot agent's path
    plot.plot(x_coord, y_coord, lw = 4.0)

    # plot agent's final position
    plot.scatter(x_coord[-1], y_coord[-1], label= "agent end position", c="blue", marker="s")

    # plot target position
    target_finder = TargetFinder()
    target_pos = target_finder.determine_target_position(self.env, seed)
    plot.scatter(target_pos[0] + offset, target_pos[1] + offset, label= "target position", c="red", marker="o")  
    
    # legend
    plot.legend(bbox_to_anchor=(1, 0), loc="lower left")

    return plot


#to do
 def change_colors():
     pass
    

  
 def create_plotgrid(self):
    """
    method creates 6 subplots and initiates the methods that combine the data with visual encodings.

    """
    # sampling to avoid visual clutter
    sampler = Sampler()
    sample_returns = sampler.stratified_sampling_return_values(self.returns, 30)
    sample_seeds = sampler .systematic_random_sampling(self.fail_seeds, 30)
    sample_actions = sampler .systematic_random_sampling(self.action_distribution, 30)

    # plot 1
    self.plot1.clear()
    self.create_success_rate_pie(self.plot1, self.returns)
    #print(self.returns)
    
    # plot 2 
    self.plot2.clear()
    self.create_return_plot(self.plot2, sample_returns)
       
    #plot 3 is a stacked bar chart that shows the total number of actions
    #create_stacked_barchart(plot3, action_distribution, x_labels)
    self.plot3.clear()
    
    self.create_stacked_horizontal_barchart(self.plot3, self.plot6, sample_seeds, sample_actions)
    
    # plot 4
    self.plot4.clear()
    self.create_grouped_barchart(self.plot4, sample_seeds, sample_actions)
    
    # plot 5
    self.plot5.clear()
    self.create_distance_plot(self.plot5, self.plot6, sample_seeds, sample_actions)
    
    # plot 6 
    self.plot6.clear()
    self.plot6.set_title("Trajectory", fontweight="bold")
        
    # button to change color encoding
    #axes = plt.axes([0.81, 0.000001, 0.1, 0.075])
    #bnext = Button(axes, 'Change color',color="cyan")
    #bnext.on_clicked(self.test)
    
    #plt.tight_layout()
    
    if UtilityVis().type_of_script() == 'jupyter':
        self.fig.canvas.draw()
    else:
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
 
    







def main(args):
   # load some data to visualize from csv file
   df = FileHandler().read_file(args.file_path)
   
   fail_seeds = [int(x) for x in [df.iloc[:, :1].values[i] for i in range(0,80)]]
   
   #print("fail_seeds: ", fail_seeds, "type: ", type(fail_seeds))

   fail_actions = []
   for i in range(0,80):
      action_list = [int(x) for x in df.iloc[i, 1].lstrip('[').rstrip(']').split(',') if x.strip().isdigit()] 
      fail_actions.append(action_list)
  
   #print("fail_actions: ", fail_actions, "type: ", type(fail_actions))

   fail_returns = [int(x) for x in [df.iloc[:, 2:3].values[i] for i in range(0,80)]]
   #print("fail_returns: ", fail_returns, "type: ", type(fail_returns))

   # test data 
   returns = [0.971875,0.971875,0.6484375,0.915625,0.915625,0.8875,0.0,0.9296875,0.915625,0.8875,0.9578125,0.915625,
             0.94375,0.0,0.8734375,0.94375,0.94375,0.9578125,0.94375,0.9578125,0.915625,0.9296875,0.971875,0.9296875,0.971875,0.88]

   returns2 = [0.94375,0.83125,0.971875,0.7890625,0.9296875,
   0.9015625,0.8875,0.9015625,0.915625,0.915625,0.8875,0.71875,0.0,0.0,0.94375,0.859375,0.915625,0.803125,0.0,0.9296875,0.915625,0.971875,0.8875,
   0.9296875,0.8734375,0.971875,0.94375,0.859375,0.9296875,0.9296875,0.8734375,0.94375,0.0,0.971875,0.94375,0.8734375,0.971875,0.94375,0.8734375,
   0.6765625,0.9296875,0.9296875,0.6625,0.94375,0.915625,0.8875,0.9296875,0.690625,0.9015625,0.915625,0.9296875,0.8734375,0.94375,0.9015625,0.8453125,
   0.83125,0.7890625,0.9015625,0.9296875,0.9578125,0.9015625,0.0,0.9296875,0.8171875,0.8734375,0.9015625,0.94375,0.7890625,0.8453125,0.94375,0.0,
   0.9296875,0.9296875,0.71875,0.9015625,0.9015625,0.803125,0.915625,0.94375,0.8734375,0.915625,0.9015625,0.8734375,0.9296875,0.8171875,0.915625,0.94375,
   0.9578125,0.9296875,0.94375,0.94375,0.8453125,0.746875,0.8734375,0.859375,0.9296875,0.94375,0.94375,0.746875,0.915625,0.9578125,0.6203125,0.94375,
   0.915625,0.94375,0.7609375,0.9296875,0.9296875,0.0,0.9296875,0.94375,0.9578125,0.8875,0.9578125,0.94375,0.9578125,0.94375,0.8734375,0.9015625,0.971875,
   0.94375,0.94375,0.915625,0.9578125,0.9578125,0.859375,0.859375,0.7328125,0.971875,0.8875,0.0,0.8734375,0.9578125,0.8453125,0.8875,0.8875,0.8453125,
   0.9578125,0.0,0.775,0.9578125,0.915625,0.9015625,0.8453125,0.915625,0.915625,0.859375,0.9578125,0.94375,0.9296875,0.7328125,0.915625,0.8171875,0.0,
   0.746875,0.8875,0.9296875,0.9859375,0.859375,0.7890625,0.8734375,0.94375,0.9015625,0.9859375,0.971875,0.9578125,0.971875,0.8453125,0.9859375,0.0,
   0.8734375,0.8171875,0.803125,0.9296875,0.9859375,0.9578125,0.8171875,0.9578125,0.71875,0.94375,0.971875,0.915625,0.8734375,0.9578125,0.9296875,0.859375,
   0.8875,0.94375,0.971875,0.9296875,0.7609375,0.9859375,0.9859375,0.859375,0.971875,0.94375,0.9296875,0.746875,0.8453125,0.94375,0.94375,0.9859375,0.94375,
   0.9578125,0.915625,0.9578125,0.9296875,0.859375,0.915625,0.971875,0.8875,0.971875,0.9296875,0.8453125,0.9578125,0.0,0.0,0.915625,0.9578125,0.94375,
   0.859375,0.8453125,0.8453125,0.915625,0.94375,0.83125,0.9296875,0.803125,0.8171875,0.8734375,0.0,0.94375,0.915625,0.9578125,0.7328125,0.9296875,0.94375,
   0.775,0.803125,0.9296875,0.9015625,0.9296875,0.859375,0.775,0.71875,0.8734375,0.8734375,0.9578125,0.8453125,0.971875,0.915625,0.859375,0.94375,
   0.9578125,0.775,0.971875,0.94375,0.0,0.746875,0.9578125,0.7890625,0.9296875,0.9859375,0.9859375,0.971875,0.7890625,0.8875,0.859375,0.8734375,0.60625,
   0.9578125,0.94375,0.0,0.9578125,0.0,0.9296875,0.8734375,0.915625,0.7890625,0.8875,0.859375,0.9578125,0.9578125,0.915625,0.9578125,0.9015625,0.8734375,
   0.9296875,0.9296875,0.9578125,0.94375,0.9015625,0.915625,0.8875,0.7890625,0.971875,0.7609375,0.0,0.71875,0.971875,0.9578125,0.94375,0.94375,0.9578125,
   0.9296875,0.94375,0.9578125,0.9015625,0.8734375,0.94375,0.7328125,0.9578125,0.9296875,0.915625,0.6625,0.971875,0.915625,0.9578125,0.9578125,0.9015625,
   0.9859375,0.8875,0.94375,0.9296875,0.746875,0.8171875,0.8875,0.0,0.8734375,0.8734375,0.9015625,0.94375,0.971875,0.9296875,0.9296875,0.7609375,0.94375,
   0.94375,0.915625,0.94375,0.6625,0.8453125,0.971875,0.8875,0.9296875,0.690625,0.971875,0.9015625,0.94375,0.94375,0.8171875,0.0,0.7890625,0.0,0.9015625,
   0.803125,0.971875,0.9015625,0.94375,0.9015625,0.9296875,0.8734375,0.94375,0.9578125,0.0,0.6203125,0.9296875,0.8875,0.8734375,0.94375,0.94375,0.8875,
   0.9578125,0.9578125,0.0,0.9578125,0.915625,0.0,0.9015625,0.803125,0.0,0.859375,0.8875,0.915625,0.971875,0.9578125,0.0,0.8875,0.94375,0.0,0.9296875,
   0.9578125,0.8453125,0.8875,0.971875,0.7328125,0.915625,0.9296875,0.9296875,0.8875,0.971875,0.9578125,0.8453125,0.9015625,0.9578125,0.859375,0.0,
   0.7609375,0.803125,0.8875,0.9015625,0.0,0.83125,0.915625,0.71875,0.9296875,0.9578125,0.8734375,0.0,0.8734375,0.8875,0.9296875,0.9015625,0.915625,
   0.9015625,0.8734375,0.859375,0.9578125,0.9015625,0.9296875,0.94375,0.9296875,0.94375,0.8734375,0.9296875,0.915625,0.94375,0.859375,0.8734375,0.9015625]

   
   plotGrid = PlotGrid()
   plotGrid.set_plotdata(args.env, fail_seeds, returns2, fail_actions)
   plotGrid.create_plotgrid()
  
   if UtilityVis().type_of_script() != 'jupyter':
       plt.show(block=True)
 


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

