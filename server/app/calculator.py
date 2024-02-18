import statistics
import gym
from scipy.stats import entropy


class Calculator():

  def calculate_weight_info(self, index, zero, nonzero):
    """
    method calculates the percentage of how many weights were changed during a backward pass.
   
    :param index: episode number selected by user.
    :param zero: number of how many gradient entries are zero.
    :param nonzero: number of how many gradient entries are not zero.
    :return: tuple with percentages of how many weights changed and how many remained unchanged.
    """
    changed = nonzero[int(index)]
    unchanged = zero[int(index)]
    changed_percentage = round(float((changed / (changed + unchanged)) * 100), 2)
    unchanged_percentage = round(float((unchanged / (changed + unchanged)) * 100), 2)

    return (changed_percentage, unchanged_percentage)


  
  def calculate_return(self, env_name, episodes, actions, seeds):
    """
    method calculates the return value of an episode. It calculates a finite-horizon undiscounded reward.

    :param env_name: BabyAI level name.
    :param episodes: number of episodes for which the method calculates the returns. 
    :param actions: actions per episode.
    :param seeds: seeds of episodes.
    :return: list of return values.
    """  
    return_values = []
    env = gym.make(env_name)
    for i in range(len(episodes)):
       return_per_episode = 0
       actions_per_episode = actions[i]
       seed_per_episode = seeds[i]
       gym.Env.seed(env, seed_per_episode)
       env.seed(seed=seed_per_episode)
       env.reset()
       for action in actions_per_episode:
         _, reward, _, _ = env.step(action)
         return_per_episode += reward

       return_values.append(return_per_episode)
    return return_values



  def calculate_missing_seeds(self, list, final_length):
   """
   method fills a list upto a specified length. 
   It fills the new list with the same values that the short list already contains. 
   It replicates the short list upto the specified length.

   :param list: original list that has an insufficient length.
   :param final_length: length of final list.
   :return: list with the specified length.
   """   
   new_list = []
   for i in range(final_length):
      j = i % len(list)
      element = list[j]
      new_list.append(element)
  
   return new_list


  def calculate_probabilities_true(self, action_true):
    """
    method calculates the probability distribution for an action specified in the trainingset.
    The specified action has probability one. All other actions have probability zero.

    :param action_true: action in training set.
    :return: probability distribution.
    """
    # list of length 7 because BabyAI action space contains 7 action.
    probabilities_true = [0] * 7
    probabilities_true[action_true] = 1
    return probabilities_true


  def calculate_entropy(self, probabilities):
    """
    method calculates the entropy.

    :param probabilities: probability distribution.
    :return: entropy of probability distribution.
    """
    return entropy(probabilities, base=2)



  def calculate_KL_divergence(self, p, q):
    """
    method calculates the Kullback-Leibler divergence. The result has 2 positions after decimal point.

    :param p: correct probability distribution.
    :param q: predicted probability distribution.
    :return: KL divergence.
    """
    return round(entropy(p, q, base=2), 2)



  def average_KLD_per_trajectory(self, indices_per_trajectory, KLD_list):
    """
    method calculates the average of KL divergence values that belong to the same trajectory.
    It rounds each average result to 2 positions after decimal point.

    :param indices_per_trajectory: nested list: the inner lists group the steps trajectory wise.
                                   Each inner list contains indices of steps that belong to the same trajectory.
    :param KLD_list: list that contains a KL divergence value for each step.
    :return: list that contains average KL divergence values.
    """
    
    KLD_grouped_trajectorywise = [KLD_list[l[0]:l[-1]+1] for l in indices_per_trajectory]
    KLD_mean = [statistics.mean(list)  if len(list) > 1 else list[0] for list in KLD_grouped_trajectorywise]
    KLD_mean = map(lambda x: round(x, 2), KLD_mean)
    return KLD_mean



  def calculate_JS_divergence(self, p, q):
    """
    method calculates the Jensen-Shannon divergence. 

    :param p: correct probability distribution.
    :param q: predicted probability distribution.
    :return: JS divergence.
    """
    m = map(lambda p_i, q_i: 0.5 * (p_i + q_i), p, q)
    result = 0.5 * self.calculate_KL_divergence(p, m) + 0.5 * self.calculate_KL_divergence(q, m)
    return result



  def calculate_divergence_list(self, probabilities_true, probabilities_predicted, KLD=True, JSD=False):
    """
    method calculates the KL or JS divergence for a list of probability distribution.

    :param probabilities_true: list of correct probability distributions given in training set.
    :param probabilities_predicted: list of predicted probability distributions.
    :param KLD: boolean; if it is true, the KL divergence is calculated.
    :param JSD: boolean; if it is true, the JS divergence is calculated.
    :return: list that contains the divergence values for all probability distributions.
    """
    results = []
    if len(probabilities_true) != len(probabilities_predicted):
         print("cannot calculate KL or JS divergence because p and q have not the same length")
    else:    
      for i, p in enumerate(probabilities_true):
        if KLD:
          results.append(self.calculate_KL_divergence(p, probabilities_predicted[i]))
        elif JSD:
          results.append(self.calculate_JS_divergence(p, probabilities_predicted[i]))
    
    return results

     

  
  