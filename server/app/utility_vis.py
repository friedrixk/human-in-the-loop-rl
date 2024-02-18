

class UtilityVis():

  def type_of_script(self):
    """
    method checks whether the program runs in a jupyter or in an ipython environment.

    :return: returns either 'jupyter' or 'ipython' 
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'



  def set_window_title(self, fig, topic):
      """
      the method places a title in the upper left corner of a matplotlib window.
      The title shows which aspect is explained.

      :param fig: figure that gets a title.
      :param topic: the topic that the visualization/ the figure explains (e.g. uncertainty).
      """
      fig.canvas.set_window_title(''.join(("Explaining ",
                                           str(topic))))



  def set_title(self, fig, level, no_episodes):
      """
      the method places a title for a matplotlib figure.
      The matplotlib figure visualizes training data. So, the title reveals 
      which level was trained and the number of episodes.
      
      :param fig: figure that gets a title.
      :param level: BabyAI level name.
      :param no_episodes: number of how many episodes build the training set.
      """
      level_name = level.split("-")[1] 
      title = ''.join((" Training " + str(level_name),
                        " with " + str(no_episodes) + " Episodes"))
      fig.suptitle(title, fontweight="bold", fontsize=16)