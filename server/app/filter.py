from app.calculator import Calculator


class Filter():
    
  def filter_episodes(self, env_name, episodes, actions, seeds):
    """
    method triggers return value calculation. It groups the episodes. 
    All failed episodes are stored in a list. All successful episodes are stored in another list. 

    :param env_name: BabyAI level name.
    :param episodes: number of episodes for which the Calculator-class calculates the returns. 
    :param actions: actions per episode.
    :param seeds: seeds of episodes.
    :return: tripel that contains the list of failed episodes, the list of successful episodes, 
            and the list of all episodes.
    """
    return_list = Calculator().calculate_return(env_name, episodes, actions, seeds)
          
    episodes_fail    = [i for i, r in enumerate(return_list) if r <= 0]   
    episodes_success = [i for i, r in enumerate(return_list) if r > 0]
    all_episodes     = [i for i, _ in enumerate(return_list)]

    return (episodes_fail, episodes_success, all_episodes)



  def filter_steps(self, episodes_fail, episodes_success, all_episodes, indices):
    """
    method filters a list of indices. Indices encode steps of an episode.
    It groups all indices that belong to successful episodes in the one list.
    Then, it groups all indices that belong to failed episodes in the one list.
    At last, it creates a list that stores all steps.

    :param episodes_fail: list that contains episode indices of failed episodes.
    :param episodes_success: list that contains episode indices of successful episodes.
    :param all_episodes: list that contains all episode indices.
    :param indices: nested list that groups step indices episode wise; 
                    each inner list contains the indices of a single episode.
    :return: tripel that contains the three created lists. 
    """

    x_fail, x_success, x_all = ([] for _ in range(3))
    for i in episodes_fail:
        for j in indices[i]:
            x_fail.append(j)

    for i in episodes_success:
        for j in indices[i]:
            x_success.append(j)

    for i in all_episodes:
        for j in indices[i]:
            x_all.append(j)
    
    return (x_fail, x_success, x_all)
