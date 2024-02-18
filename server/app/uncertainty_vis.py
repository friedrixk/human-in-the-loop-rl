from argparse import ArgumentParser
import matplotlib
from matplotlib import pyplot as plt
from IPython import display
from matplotlib.collections import LineCollection
from matplotlib.widgets import RangeSlider
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from app.utility_vis import UtilityVis
from app.color import Color
from app.trajectory import Trajectory
from app.sampler import Sampler
from app.file_handler import FileHandler
from app.filter import Filter
from app.calculator import Calculator
from app.interactive_features import InteractiveFeatures



parser = ArgumentParser()
parser.add_argument("--file_path", required=True, help="specify path to file that stores the visualization data")
parser.add_argument("--file_seed_list", required=True, help="specify path to file that stores the seeds used in demo generation")
parser.add_argument("--save_path", required=True)


class UncertaintyVis():

 def __init__(self):

    #matplotlib.style.use("Solarize_Light2")
    #matplotlib.style.use("ggplot")
    matplotlib.style.use("seaborn-dark")
    
    self.fig = plt.figure(1, figsize=(9.5, 9.5), dpi=120)

    # figure 2 is used to create glyphs
    #self.fig2 = plt.figure(2)
    #plt.figure(2)
    #plt.close()
    plt.figure(1)
    
    self.plot1 = plt.subplot2grid((350,350), (0, 0), rowspan=156, colspan=230)
    self.plot2 = plt.subplot2grid((350,350), (0, 240), rowspan=156, colspan=110, sharey=self.plot1)
    self.plot3 = plt.subplot2grid((350,350), (195, 0), rowspan=155, colspan=180)
    self.plot4 = plt.subplot2grid((350,350), (205, 220), rowspan=155, colspan=155)
    plt.tight_layout()

    # workaround matplotlib/ipympl #290
    if UtilityVis().type_of_script() == 'jupyter':
        display.display(self.fig.canvas)
        self.fig.canvas._handle_message(self.fig.canvas, {'type': 'send_image_mode'}, [])
        self.fig.canvas._handle_message(self.fig.canvas, {'type': 'refresh'}, [])
        self.fig.canvas._handle_message(self.fig.canvas, {'type': 'initialized'}, [])
        self.fig.canvas._handle_message(self.fig.canvas, {'type': 'draw'}, [])


    
 def set_plotdata(self, env_name, seeds, actions_true, probabilities, probabilities_true, 
                 step_indices_per_trajectory, trajectory_predicted, trajectory_true, KLD,
                 x_fail, x_success, x_all):
     self.env_name = env_name
     self.seeds = seeds
     self.actions = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
     self.actions_true = actions_true
     self.probabilities_true = probabilities_true
     self.probabilities = probabilities
     self.indices_per_trajectory = step_indices_per_trajectory
     self.trajectory_predicted = trajectory_predicted
     self.trajectory_true = trajectory_true
     self.KLD = KLD
     self.selection = KLD
     self.last_selection = None
     self.current_step = 0
     self.current_xrange = np.arange(0, len(self.KLD), 1)
     self.current_yrange = np.arange(0, max(self.KLD)+0.2, 0.1)
     self.x_fail = x_fail
     self.x_success = x_success
     self.x_all = x_all
     self.x_filtered = x_all
    

 
 def filter_steps(self, event):
     """
     If the user clicks a filter button, then the click invokes this method.
     The method updates the class variable x_filtered. It stores the steps that fulfill the filter critrion.
     Steps are filtered based on whether they belong to a successful or failed episode.

     :param event: button click event.
     """
     
     # all episodes, no filtering
     if axes_all_eps.in_axes(event):
        print("show all")
        self.x_filtered = self.x_all
        
     # button to show only successful episodes was selected
     elif axes_success_eps.in_axes(event):
        print("filter successful episodes")
        self.x_filtered = self.x_success

     # button to show only failed episodes was selected  
     elif axes_fail_eps.in_axes(event):
        print("filter failed episodes")
        self.x_filtered = self.x_fail
       

     # select points on x and y axes that fulfill filter criterion
     # x axis maximum is set to 100 points
     self.x_filtered = list(filter(lambda i: i < 100, self.x_filtered))
     KLD_filtered = [self.KLD[i] for i in self.x_filtered]
    
     # update display
     self.create_multicolored_line(self.plot1, KLD_filtered, self.x_filtered, "KL divergence", self.KLD, self.x_all[0:len(self.KLD)])
     self.mark_epsisodes(self.plot1)
     self.create_histogram(self.plot2, KLD_filtered)
     slider_x.set_val([0, len(self.KLD)])
     slider_y.set_val([0, max(self.KLD)])

     

 def on_click(self, event):
     """
     method is invoked whenever the user clicks in the uncertainty line plot. It updates the probability distribution plot
     and the trajectory plot. After a click, the plot shows information that is related with the clicked point.

     :param event: mouse click event.
     """
     threshold = 0.8
     mouse_x = round(event.xdata)
     
     if (self.plot1.in_axes(event) and event.ydata >= self.KLD[mouse_x] - threshold and
         event.ydata <= self.KLD[mouse_x] + threshold):
        
        # calculate relative x value 
        # if the user adjust the slider, the absolute value does not correspond to correct data point
        # mouse_x value is an absolute value that does not correspond to desired x value in the filtered view
        zoom_factor = (self.current_xrange[-1] - self.current_xrange[0]) / (len(self.KLD)-1)
        x = int(self.current_xrange[0] + (mouse_x * zoom_factor ))
        
        # update label that indicates which point was selected
        if self.last_selection is not None:
            self.last_selection.remove()

        self.last_selection = InteractiveFeatures().highlight_selected_point(self.plot1, (x, self.KLD[x]),  5.5)

        self.current_step = x

        # get color for bars in probability distribution chart
        color = Color().get_cmap_color(self.KLD, self.KLD[x])

        # update plots with new data
        self.create_probability_distr(self.plot3, self.current_step, self.probabilities[x], self.actions, 
                                      self.probabilities_true[x], self.KLD[x], color)
        self.create_trajectory(self.plot4, x, color)

        # redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

  

 def step(self, event):
     """
     the step buttons invoke this method. The method determines whether the user clicked on 
     the step forward or step backward button. It updates all necessary data. 
     Then, it triggers either to display the next step or the previous step. 

     :param event: button click event.
     """
     # step backward button clicked
     if axes_stepb.in_axes(event):
       self.current_step -= 1

       if self.current_step < 0:
        self.current_step = len(self.KLD) - 1 
     
     # step forward button clicked
     if axes_stepf.in_axes(event):
        self.current_step += 1

        if self.current_step > len(self.KLD) - 1:
          self.current_step = 0

     
     # update the "selected" label
     if self.last_selection is not None:
        self.last_selection.remove()
        self.last_selection = None

     self.last_selection = InteractiveFeatures().highlight_selected_point(self.plot1, (self.current_step, self.KLD[self.current_step]),  5.5)
   
     # get color for bars in probability distribution chart
     color = Color().get_cmap_color(self.KLD, self.KLD[self.current_step])

     # update plots with new data
     self.create_probability_distr(self.plot3, self.current_step, self.probabilities[self.current_step], self.actions, 
                                    self.probabilities_true[self.current_step], self.KLD[self.current_step], color)
     self.create_trajectory(self.plot4, self.current_step, color)

     # redraw the figure to ensure it updates
     self.fig.canvas.draw_idle()



 def update_y(self, val):
    """
    method updates the main plot whenever the user selects another range in the vertical slider.

    :param val: new y range selected by the user.
    """
    # select new data    
    KLD_selection = [self.KLD[i] for i in self.x_filtered if i in self.current_xrange]
    self.selection =  list(filter(lambda k: k > val[0] and k < val[1], KLD_selection))
    steps = [i for i, kld in enumerate(self.KLD) if (kld in self.selection and i in self.x_filtered and i in self.current_xrange)] 
    
    self.current_yrange = np.arange(val[0], val[1], 0.1)
    
    # display new data
    self.update_display(steps)

   

 def update_x(self, val):
    """
    method updates the main plot whenever the user selects another range in the horizontal slider.

    :param val: new x range selected by the user.
    """
    # select new data
    self.current_xrange = np.arange(int(val[0]), int(val[1])+1, 1)
   
    KLD_selection = [self.KLD[i] for i in self.x_filtered if i in self.current_xrange]
    self.selection =  list(filter(lambda k: k < self.current_yrange[-1] and k > self.current_yrange[0], KLD_selection))
    steps = [i for i, kld in enumerate(self.KLD) if (kld in self.selection and i in self.x_filtered and i in self.current_xrange) ] 
    
    # display new data
    self.update_display(steps)

  

 def update_display(self, steps):
    """
    method triggers a display update. 
    After the update, the plots show data that the user has filtered via a slider.

    :param steps: step indices that fullfill the filter criteria.
    """
    # visualize selected data
    self.create_multicolored_line(self.plot1, self.selection, steps, "KL divergence", self.KLD, self.x_all)
    self.mark_epsisodes(self.plot1)
    self.create_histogram(self.plot2, self.selection)

    self.last_selection = InteractiveFeatures().highlight_selected_point(self.plot1, (self.current_step, self.KLD[self.current_step]),  5.5)

    # redraw the figure to ensure it updates
    self.fig.canvas.draw_idle()
     


 def create_multicolored_line(self, plot, uncertainty_values, x, label, all_KLD, all_x):
    """
    method creates a line chart. Some data points are marked with dots. 
    The dot color varies depending on the uncertainty value.
    The method adds a tooltip for the line.

    :param plot: plot where the line appears.
    :param uncertainty_values: agent's uncertainty values; encoded by y value and color.
    :param x: the x values.
    :param label: y axis label (either KL divergence or JS divergence).
    :param all_KLD:
    :param all_x:
    """
    #print("x: ", x, "uncertainty: ", uncertainty_values)
    plot.clear()
    plot.set_xlim(self.current_xrange[0], self.current_xrange[-1])
    plot.set_ylim(0, max(self.KLD) + 1)
    plot.grid()
    
    
    # draw a line collection
    all_x = self.x_all[0:len(self.KLD)]
    all_KLD = self.KLD
    points = np.array([all_x, all_KLD]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(min(self.KLD), max(self.KLD))
    lc = LineCollection(segments, lw=2, cmap="cool", norm=norm, colors="tab:blue", zorder=9.0)
    lc.set_array(all_KLD)
    lc.set_linewidth(2)
    line = plot.add_collection(lc)

    # color data points according to uncertainty value
    colors = [Color().get_cmap_color(self.KLD, val) for val in uncertainty_values]
    plot.scatter(x, uncertainty_values, c=colors, marker="o", zorder=10.0) 

    # color points that do not fulfill the filter criteria in grey
    x1 = [item for item in self.current_xrange if item not in self.x_filtered or item not in x]
    y1 = [self.KLD[i] for i in x1 if i < len(self.KLD)-1]
    plot.scatter(x1[0:len(y1)], y1, c="lightgrey", marker="o", zorder=10.0)
      
    # place color bar
    axes_colorbar = plt.axes([0.065, 0.468, 0.18, 0.478])
    axes_colorbar.set_visible(False)
    self.fig.colorbar(line, ax=axes_colorbar, location="left", ticks=[], label=label, shrink=0.75)
    
    # tooltip 
    InteractiveFeatures().add_tooltip_uncertainty(plot)
    

    
    


 def mark_epsisodes(self, plot):
    """
    the method draws vertical lines. Each line marks an episode end.
    All points between two lines belong to one episode.
    Additionally, the method places a second x axis at the top.

    :param plot: plot in which the lines appear.
    """
    x_axis_top = plot.secondary_xaxis("top")
    x_axis_top.set_xlabel("epsiodes", loc="right", c="lightgrey", zorder=9.0)

    labels, x_ticks, y,  = [], [], [0, max(self.KLD) + 1.0]
   
    for i, step_indices in enumerate(self.indices_per_trajectory):
     # the x axis shows at most 100 points to avoid visual clutter
     if step_indices[-1] > 100:
         break

     if step_indices[-1] in self.current_xrange:
        labels.append(i)
        x_ticks.append(step_indices[-1])
        plot.plot([step_indices[-1], step_indices[-1]], y, c="lightgrey")
        
    x_axis_top.set_xticks(x_ticks)
    x_axis_top.set_xticklabels(labels, c="lightgrey", zorder=5.0)
    


 def create_histogram(self, plot, uncertainty_values):
    """
    method creates a horizontal histogram. The uncertainty values are aligned on the y axis.
    The bar's height (x value) shows how often an uncertainty value appears in the set of uncertainty values.
    The bar's color encodes the uncertainty value.
    The method adds a tooltip as well.

    :param plot: plot where the histogram appears.
    :param uncertainty_values: agent's uncertainty values.
    """
    plot.clear()
    plot.set_title("Histogram of KL Divergence")
    plot.set_xlabel("incidence", loc="right")
    plot.set_ylim(0, max(self.KLD) +0.2)
    plot.grid()
   
    bar_width = 0.05
  
    x = []
    y_range = np.arange(0, max(self.KLD)+0.2, 0.1)

    # count how often a value appears in the line plot
    for i in y_range:
        x.append(len(list(filter(lambda x: float(round(x, 1)) == float(round(i, 1)), uncertainty_values))))
    
    colors = [Color().get_cmap_color(self.KLD, val) for val in y_range]
    
    plot.barh(y_range, x, bar_width, color=colors)

    # tooltip
    InteractiveFeatures().add_tooltip_standard(plot)



 def create_probability_distr(self, plot, step_no, probabilities, actions, probabilities_true, KLD, color):
     """
     the method creates a bar chart. The BabyAI action space is aligned on the x axis. 
     The probability range from 0 to 1 is aligned on the y axis.
     The bar's height encodes the probability of an action. The bar's color encodes the uncertainty value.
     
     :param plot: plot where the bar chart appears.
     :param step_no: number of step that the user has selected.
     :param probabilities: 
     :param actions: action space.
     :param probabilities_true:
     :param KLD: KL divergence between predicted and true probability distribution.
     :param color: bar's color that encodes the KL divergence.
     """
     plot.clear()
     plot.set_title("Probability distribution of step " + str(step_no), loc="left")
     plot.set_xlabel("actions", loc="right")
     plot.set_ylabel("probability")
     plot.set_ylim(0, 1.05)
     plot.grid()

     bar_width = 0.5     
     
     bars_true = plot.bar(actions, probabilities_true, bar_width, color="yellowgreen", alpha=0.5, label="true action")
     bars = plot.bar(actions, probabilities, bar_width, color=color, alpha=1.0, label="network prediction")
     
     # add percentage above each bar
     annotations = InteractiveFeatures().annotate_percentage(plot, bars, 1/100, 8, (0, 9))

     # legend
     leg = plot.legend(bbox_to_anchor=(1, 0), loc="lower left", fancybox=True, frameon=True)

     #tooltip 
     InteractiveFeatures().add_tooltip_uncertainty(plot, probabilities, actions, KLD)

     


 def create_trajectory(self, plot, step_no, color):
    """
    method determines to which trajectory the selected step belongs.
    It triggers the trajectory visualization. The visualization shows the optimal trajectory and 
    the predicted trajectory. Moreover, it highlights the current step.  

    :param plot: plot where the trajectory appears.
    :param step_no: number of step that is highlighted in trajectory.
    :param color: color for highlighting the step. Color encodes the uncertainty value.
    """
    
    actions, actions_true = [], []
    seed = None
    for i, idx_list in enumerate(self.indices_per_trajectory):
        if step_no in idx_list:
            step_idx = idx_list.index(step_no)
            actions = self.trajectory_predicted[i]
            actions_true = self.trajectory_true[i]
            seed = self.seeds[i]
            break
    
    if actions and actions_true and (seed is not None):
       trajectory.visualize_trajectory(self.env_name, plot, actions, actions_true, seed, step_no, True, step_idx, color)
    else:
        print("step cannot be mapped to trajectory")

  


 def create_plotgrid(self, file_handler):
     """
     method arranges all plots in a window. It adds range slider and buttons. 

     :param file_handler: file handler saves the figure in a file.
     """
     global trajectory
     trajectory = Trajectory()

     x = np.arange(0, len(self.KLD), 1)

     self.create_multicolored_line(self.plot1, self.KLD, x, "KL divergence", self.KLD, x)
     self.mark_epsisodes(self.plot1)
     self.create_histogram(self.plot2, self.KLD)
     InteractiveFeatures().add_explanation([self.plot3, self.plot4],
                            "Click on a point of the line above \nto see the corresponding data")
     
     
     # add horizontal slider 
     global slider_axes_x, slider_x
     slider_axes_x = plt.axes([0.15, 0.49, 0.46, 0.02])
     slider_x = RangeSlider(slider_axes_x, "Select steps", 0, len(self.KLD), valinit=(0, len(self.KLD)), orientation="horizontal")
     slider_x.on_changed(self.update_x)

     # add vertical slider 
     global slider_axes, slider_y
     slider_axes_y = plt.axes([0.09, 0.53, 0.015, 0.35])
     slider_y = RangeSlider(slider_axes_y, "Select range", 0, max(self.KLD), valinit=(0, max(self.KLD)), orientation="vertical")
     slider_y.on_changed(self.update_y)

     # button
     global axes_stepf, axes_stepb, axes_fail_eps, axes_success_eps, axes_all_eps
    
     axes_save        = plt.axes([0.92, 0.000002, 0.08, 0.08])
     axes_rbtn        = plt.axes([0.85, 0.35, 0.08, 0.08])
     axes_stepf       = plt.axes([0.512, 0.18, 0.08, 0.08])
     axes_stepb       = plt.axes([0.52, 0.25, 0.062, 0.06])
     axes_all_eps     = plt.axes([0.93, 0.69, 0.065, 0.06])
     axes_fail_eps    = plt.axes([0.93, 0.76, 0.065, 0.06])
     axes_success_eps = plt.axes([0.93, 0.83, 0.065, 0.06])
     
     # button images
     icon_save = plt.imread("scripts/save_button.jpg") #221021 fred: changed "scripts\\save_button" to "scripts/save_button"
     icon_next = plt.imread("scripts/next_button.jpg") #221021 fred: changed "scripts\\next_button" to "scripts/next_button"
     icon_back = plt.imread("scripts/backButton.jpg") #221021 fred: changed "scripts\\backButton" to "scripts/backButton"
     #axes_KLD = plt.axes([0.005, 0.62, 0.08, 0.072])
     #axes_JSD = plt.axes([0.005, 0.53, 0.08, 0.072])

     global btn_save, btn_KLD, btn_JSD, btn_step_forward, btn_step_backward, btn_fail_eps, btn_success_eps, btn_all_eps
     global btn_trajectory
     btn_save = Button(axes_save, label="", color="white", hovercolor="cadetblue", image=icon_save)
     btn_save.on_clicked(file_handler.save)


     btn_trajectory = RadioButtons(axes_rbtn, ("both paths", "optimal path", "agent path"))
     btn_trajectory.on_clicked(lambda label: trajectory.set_type(label))
     # InteractiveFeatures().add_button(axes_save, label="", file_handler.save, image=icon_save)
     #btn_KLD = InteractiveFeatures().add_button(axes_KLD, "show KLD", print("not implemented"))
     #btn_JSD = InteractiveFeatures().add_button(axes_JSD, "show JSD", print("not implemented"))
               
     #btn_KLD = Button(axes_show_KLD, "show KLD", color="powderblue", hovercolor="cadetblue")
     #btn_KLD.on_clicked(print("not implemented yet"))

     #btn_JSD = Button(axes_show_JSD, "show JSD", color="powderblue", hovercolor="cadetblue")
     #btn_KLD.on_clicked(print("not implemented yet"))
     
     btn_step_forward = Button(axes_stepf, label="", color="powderblue", hovercolor="cadetblue", image=icon_next)
     btn_step_forward.on_clicked(self.step)

     btn_step_backward = Button(axes_stepb, label="", color="powderblue", hovercolor="cadetblue", image=icon_back)
     btn_step_backward.on_clicked(self.step)

     btn_all_eps = Button(axes_all_eps, "show all\n episodes", color="powderblue", hovercolor="cadetblue")
     btn_all_eps.on_clicked(self.filter_steps)

     btn_fail_eps = Button(axes_fail_eps, "show fail\n episodes", color="powderblue", hovercolor="cadetblue")
     btn_fail_eps.on_clicked(self.filter_steps)

     btn_success_eps = Button(axes_success_eps, "show successful\n episodes", color="powderblue", hovercolor="cadetblue")
     btn_success_eps.on_clicked(self.filter_steps)


     self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    



def main(args):
    file_handler = FileHandler()
    file_handler.set_path(args.save_path)


    # load some data to visualize from csv file
    df_seedlist = file_handler.read_file(args.file_seed_list)
    df = file_handler.read_file(args.file_path)

    env_name = (df_seedlist.iloc[:, 0:1].values[0])[0]
    train_demo_seeds = [int(x) for x in [df_seedlist.iloc[:, 1:].values[i] for i in range(0, df_seedlist.shape[0])]]

    # preprocess data
    sampler = Sampler()
    sample = sampler.sample_batchwise(df, len(train_demo_seeds), 130)
   
    print(sample)

    # filter predicted actions from data frame and parse the string to integer list
    a1 = sample.iloc[:, 6:7].values[0:]
    trajectory_predicted = []
    for i in  range(0, sample.shape[0]):
      a2 = a1[i, 0].lstrip('[').rstrip(']').split(',')  
      a3 = [int(x) for x in a2 if x.strip().isdigit()]
      trajectory_predicted.append(a3)
    #print("trajectories: ", trajectory_predicted)
    
    # filter true actions (specified in training demo) from data frame and parse the string to integer list
    demo_actions = sample.iloc[:, 7:8].values[0:]
    trajectory_true = []
    actions_true = []
    for i in  range(0, sample.shape[0]):
      list1 = demo_actions[i, 0].lstrip('[').rstrip(']').split(',')  
      list2 = [int(x) for x in list1 if x.strip().isdigit()]
      trajectory_true.append(list2)
      # store actions in single list instead of nested list
      for l2 in list2:
          actions_true.append(l2)
    #print("true actions: ", actions_true)
    #print("true trajectories: ", trajectory_true)
    

    # sort step indices that belong to same trajectory into one list
    # necessary for mapping step number to corresponding trajectory when creating trajectory plot 
    step_indices_per_trajectory = []
    offset = 0
    for trajectory in trajectory_true:
        indices = list(np.arange(offset, offset+len(trajectory), 1))
        step_indices_per_trajectory.append(indices)
        offset += len(trajectory)
    #print(trajectory_step_indices)


    seed_list_enlarged = Calculator().calculate_missing_seeds(train_demo_seeds, len(trajectory_predicted))

    # filter probabilites predicted by neural network 
    predictions = sample.iloc[:, 8:9].values[0:]
    pred_probabilities = []
    for i in  range(0, sample.shape[0]):
      p1 = predictions[i, 0].lstrip('[').rstrip(']')
      #print(p1, type(p1)) 
      p2 = p1.split('[')
      p3 = [s.rstrip('], ') for s in p2]
      #print(p2, type(p2)) 
      #print(p3, len(p3)) 

      for string in p3:
          string = string.split(',')
          
          # probab_list contains probability distribution for one step
          probab_list = [round(float(s), 4) for s in string]
          pred_probabilities.append(probab_list)
    #print("pred_probabilities: ", pred_probabilities)
   

    # calculate probability distribution for each action contained in training demo set
    probabilities_true = []
    l = list(map(lambda a : Calculator().calculate_probabilities_true(a), actions_true)) 
    for i, l1 in enumerate(l):
        probabilities_true.append(l1)
    #print("probabilities_true: ", probabilities_true)
    
    uncertainty_vis = UncertaintyVis()
    KLD = Calculator().calculate_divergence_list(probabilities_true[0:100], pred_probabilities[0:100])
 
    x = np.arange(0, len(KLD), 1)
    (episodes_fail, episodes_success, all_episodes) = Filter().filter_episodes(env_name, x, trajectory_predicted, seed_list_enlarged)
    (x_fail, x_success, x_all) = Filter().filter_steps(episodes_fail, episodes_success, all_episodes, step_indices_per_trajectory)
    
    UtilityVis().set_window_title(uncertainty_vis.fig, "Uncertainty")
    UtilityVis().set_title(uncertainty_vis.fig, env_name, len(train_demo_seeds))
    uncertainty_vis.set_plotdata(env_name, seed_list_enlarged, actions_true, pred_probabilities[0:100], probabilities_true[0:100], 
                                step_indices_per_trajectory, trajectory_predicted, trajectory_true, KLD, x_fail, x_success, x_all)
    uncertainty_vis.create_plotgrid(file_handler)


    if UtilityVis().type_of_script() != 'jupyter':
       plt.show(block=True)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
