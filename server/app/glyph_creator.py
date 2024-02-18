import datetime
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from app.trajectory import Trajectory



class Glyph():
 def __init__(self):
    pass


 def set_glyph_max(self, max):
    """
    method sets the maximum number of glyphs on the display.

    :param max: maximum number of how many glyphs are displayed.
    """
    self.max = max



 def init_display_list(self):
     """
     method initializes or resets the three lists. The lists store the displayed glyphs and the selected episodes.
     """
     self.selected_traj = []
     self.traj_displayed = [None for _ in range(self.max)]
     self.pie_displayed = [None for _ in range(self.max)]



 def make_glyph_folder(self, glyph_type):
     """
     method creates a folder for storing glyphs. Each folder gets a glyph name and a timestamp. 
     
     :param glyph_type: specifies how the glyph looks like (e.g., a pie chart).
     :return: glyph folder's path.
     """

     time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    
     new_abs_path = "glyph_" + glyph_type + time_stamp   # create absolute path of a new folder 
     if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)

     return new_abs_path



 def save_glyph(self, path_folder, glyph_type, glyph_no):
     """
     method saves the glyph as file in the glyph folder.

     :param path_folder: glyph folder's path.
     :param glyph_type: type of glyph, e.g. pie.
     :param glyph_no: number to identify glyph and to find correct glyph for each episode.
     """
     path = path_folder + "\\" + glyph_type + "_" + str(glyph_no) + ".png"
     plt.savefig(path, bbox_inches="tight", pad_inches=0.0)



 def get_glyph_path(self, path_folder, glyph_type, glyph_no):
    """
    method finds path for a requested glyph.

    :param path_folder: glyph folder's path.
    :param glyph_type: type of glyph, e.g. pie.
    :param glyph_no: number to identify glyph and to find correct glyph for each episode.
    :return: path to requested glyph.
    """
    path = path_folder + "\\" + glyph_type + "_" + str(glyph_no) + ".png"
    return path



 def create_trajectory_glyph(self, traj_folder, env, seed, actions, actions_true, glyph_no):
     """
     method creates the trajectory plot and stores the plot in a file. The file is used to create and picture the glyphs.
     
     :param traj_folder: path to folder that stores the created glyph.
     :param env: BabyAI level name.
     :param seed: seed of training episode.
     :param actions: predicted actions of one training episode.
     :param actions_true: true actions specified in demo.
     :param glyph_no: identifier.
     """
     plt.figure(2)
     plt.clf()
     glyph_plot = plt.subplot()
     
     Trajectory().visualize_trajectory(env, glyph_plot, actions, actions_true, seed, glyph_no)
     self.save_glyph(traj_folder, "traj", glyph_no)
     plt.close()
    



 def create_pie_glyph(self, pie_folder, accuracy, glyph_no):
    """
    method creates a new pie chart glyph for accuracy value.
    It uses the method save_glyph() to store the pie chart.
    
    :param pie_folder: path to folder that stores the created glyph.
    :param accuracy: prediction accuracy.
    :param glyph_no: identifier.
    """
    plt.figure(2)
    plt.clf()
    plot = plt.subplot()
    plot.set_title("Episode: " + str(glyph_no))
    colors = ["yellowgreen", "crimson"]
    explode = (0, 0)
    text_style = {"fontsize": 13, "color":"black"}

    percentages = [accuracy * 100, (1-accuracy) * 100]
    plot.pie(percentages, explode=explode, colors=colors,
      autopct="%.0f%%", 
      pctdistance=1.15,
      radius=3.0,
      shadow=False,
      textprops=text_style,
      labeldistance=0.3,
      wedgeprops={"linewidth": 3.0, "edgecolor": "white"},)
    plot.axis("equal")
    self.save_glyph(pie_folder, "pie", glyph_no)
    
    plt.close()

    #centre_circle = plt.Circle((0,0), 0.67, color="grey", fc="white", linewidth=0.4)
    #plot.add_artist(centre_circle)



 def place_glyph(self, plot, path_folder, glyph_type, x, y):
    """
    method read the glyph image form the glyph folder's path. It places the glyph above the selected point.
     
    :param plot: the plot in which to place a new glyph.
    :param path_folder: path to folder that contains the glyphs.
    :param glyph_type: type indicates whether the glyph shows the trajectory or the accuracy.
    :param x: x position of selected point.
    :param y: y position of selected point.
    """


    coordinates = (0,0)
    x_span = plot.get_xlim()[1] - plot.get_xlim()[0]
    offset = x_span * 0.1
    y1 = plot.get_ylim()[1] - 0.5
    if glyph_type.__eq__("traj"):
      coordinates = (x + offset, y1)
    elif glyph_type.__eq__("pie"):
       coordinates = (x - offset, y1)
    else:
       return
    
    path = self.get_glyph_path(path_folder, glyph_type, x)
    glyph = mpimg.imread(path)
    imagebox = OffsetImage(glyph, zoom=0.25)
    ab = AnnotationBbox(imagebox, xy=(x,y), xybox=coordinates, annotation_clip=False,  
                        arrowprops=dict(arrowstyle="-", color="grey", clip_on=False, zorder=10.0))
    
    ab.set_zorder(10.0)
    plot.add_artist(ab)
    ab.draggable()                                   

    # update list that stores which glyphs are displayed already
    if glyph_type.__eq__("traj"):
      self.traj_displayed[x] = ab
    elif glyph_type.__eq__("pie"):
      self.pie_displayed[x] = ab

   

 def remove_glyph(self, x, glyph_type):
    """
    method removes the glyph belonging to the selected episode.

    :param x: x position of selected point.
    :param glyph_type: type of glyph to remove (e.g. pie, trajectory or all).
    """
   
    # remove trajectory glyph
    if glyph_type.__eq__("traj") or glyph_type.__eq__("all"):
      glyph = self.traj_displayed[x]
   
      if glyph is not None:
         glyph.remove()

      self.traj_displayed[x] = None
     
   
    # remove pie glyph
    #if glyph_type.__eq__("pie") or glyph_type.__eq__("all"):
      #glyph = self.pie_displayed[x]
      #self.pie_displayed[x] = None
      #if glyph != None:
         #glyph.remove()



 def remove_all_glyphs(self, glyph_type):
    """
    method removes all glyphs.

    :param glyph_type: type of glyph to remove (e.g. pie, trajectory or all).
    """
    
    print("remove all glyphs...")
    for i in range(len(self.traj_displayed)):
      self.remove_glyph(i, glyph_type)
    plt.draw()
