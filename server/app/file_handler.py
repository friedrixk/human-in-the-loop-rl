
import datetime
from matplotlib import pyplot as plt
import pandas as pd


class FileHandler():
  def set_path(self, path):
    """
    Method sets a path to the saved files.

    :param path: path that is stored in a class variable.
    """  
    self.save_to = path



  def save(self, _):
    """
    The save button invokes this method. It saves the current visualization in a file.
    The filename includes a time stamp to identify the visualization.

    :param _: button press event.
    """
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    path = self.save_to + "\\HITL_RL_Explainability_" + time_stamp + ".png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0)



  def read_file(self, path):
    """
    Method that reads a CSV file and stores the data in a data frame. 
    
    :param path: path to the CSV file.
    :return: data frame that stores the content of the CSV file.
    """
    df = pd.DataFrame()
    df = pd.read_csv(path, delimiter=",")
    return df
