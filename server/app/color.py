
from matplotlib import pyplot as plt


class Color():

    def get_cmap_color(self, norm_list, val):
     """
     method determines which color in the color map corresponds to a data value.

     :param norm_list: list used to normalize the color map. 
     :param val: data point that needs a color.
     :return: color.
     """
     cmap = plt.get_cmap("cool")
     norm = plt.Normalize(min(norm_list), max(norm_list))
     color = cmap(norm(val))
     return color



    def map_action_to_color(self, action_number):
     """
     method maps a number that encodes an action to the corresponding color of this action.

     :param action_number: integer that encodes an action in BabyAI action space.
     :return: the color that encodes the action.
     """
     if(action_number == 0):
        color = "tab:orange"
     elif(action_number == 1):
        color = "tab:cyan" 
     elif(action_number == 2):
         color = "tab:blue"
     elif(action_number == 3):
         color = "tab:olive" 
     elif(action_number == 4):
         color = "tab:purple" 
     elif(action_number == 5):
         color = "tab:grey"
     elif (action_number is None):
         color = "lightgrey"
     else:
        return "action number is not part of environment's action enumeration"
     return color