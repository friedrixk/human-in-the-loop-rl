"""
Visualize a sequence of fail actions on a given environment.
"""

import argparse
import gym
import time
import pandas as pd
from gym_minigrid.window import Window

import matplotlib.pyplot as plt
from IPython import display


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.5,
                    help="the pause between two consequent actions of an agent")
#parser.add_argument("--manual-mode", action="store_true", default=False,
 #                   help="Allows you to take control of the agent at any point of time")

parser.add_argument("--tile_size", type=int, default=32,
                    help="size at which to render tiles")
parser.add_argument('--agent_view', default=False,
                    help="draw the agent sees (partially observable view)", action='store_true')
parser.add_argument('--path', default="",
                     help="path to csv file")
parser.add_argument('--replay_no', type=int, default=1, required=True,
                     help="number which action sequence you want to watch")


def type_of_script():
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



def redraw(img):
    """
    method to update the window.

    :param img: new image to display
    """
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)


    if type_of_script() == 'jupyter':
        if window.imshow_obj is None:
             window.imshow_obj = window.ax.imshow(img, interpolation='bilinear')

        window.imshow_obj.set_data(img)
        display.clear_output(wait=True) 
        display.display(plt.gcf())
    else:
        window.show_img(img)
    



def reset():
    """
    method to reset the environment. It prints the mission and show the mission in the window.
    """
    env.seed(seed)
    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)



def step(action):
    """
    method that takes as input an action and has no return value.
    If the mission is fulfilled after the action, the method initiates the reset. Otherwise it initiates the redraw.

    :param action: agent's next action. 
    """
    #actions.append(action)
    obs, reward, done, _ = env.step(action)
    
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)


#def key_handler(event):
    #print('pressed', event.key)

    #if event.key == 'escape':
        #window.close()
        #return

    #if event.key == 'backspace':
        #reset()
        #return

    #if event.key == 'left':
        #step(env.actions.left)
        #return
    #if event.key == 'right':
        #step(env.actions.right)
        #return
    #if event.key == 'up':
        #step(env.actions.forward)
        #return

    # only for testing replay feature 
    #if event.key == 'r':
        # replay feature is only possible if some actions have been chosen beforehand
     #   if actions:
      #     replay(args.seed, actions, args.pause)
       #    actions.clear()
        #else:
         #   print("you have to specify a sequence of actions")
        #return
        
    # Spacebar 
    #if event.key == ' ':
        #step(env.actions.toggle)
        #return
    #if event.key == 'pageup' or event.key == 'p':
        #step(env.actions.pickup)
        #return
    #if event.key == 'pagedown' or event.key == 'd':
        #step(env.actions.drop)
        #return

    #if event.key == 'enter':
        #step(env.actions.done)
        #return



def action_to_string(action_number):
    """
    method maps an integer to the name/string of the corresponding action.

    :param action_number: integer that encodes an action.
    :return: name of an action, e.g., "left" or "right".
    """

    if(action_number == env.actions.left):
        return "left"
    elif(action_number == env.actions.right):
        return "right"
    elif(action_number == env.actions.forward):
        return "forward"
    elif(action_number == env.actions.toggle):
        return "toggle"
    elif(action_number == env.actions.pickup):
        return "pickup"
    elif(action_number == env.actions.drop):
        return "drop"
    elif(action_number == env.actions.done):
        return "done"
    else:
        return "action number is not part of environment's action space"



def map_number_to_action(action_number):
    """
    method that maps an integer to the corresponding action in the environment's action space.

    :param action_number: integer that encodes an action.
    :return: action in the environment's action space.
    """
    action_number = int(action_number)

    if(action_number == 0):
        return env.actions.left
    elif(action_number == 1):
        return env.actions.right
    elif(action_number == 2):
        return env.actions.forward
    elif(action_number == 3):
        return env.actions.pickup
    elif(action_number == 4):
        return env.actions.drop
    elif(action_number == 5):
        return env.actions.toggle
    elif(action_number == 6):
        return env.actions.done
    else:
        return "action number is not part of environment's action space"




def read_action_file(path):
    """
    method that reads a CSV file that stores the information needed to watch the replay of the fail episode. 
    
    :param path: path to the CSV file.
    :return: data frame that stores the content of the CSV file.
    """

    df = pd.DataFrame()
    df = pd.read_csv(path, delimiter=",")
    #df = df.dropna()
    
    seeds = df.iloc[:, :1]      # the first column of the csv file contains the fail seeds
    actions = df.iloc[:, 1:2]   # the second column contains the sequence of actions that the agent tried
    returns = df.iloc[:, 2:]    # the third column contains the return that is always zero if the agent failed

    print(df.head())
    print("seeds: \n", seeds)
    print( "actions: \n", actions)
    print( "returns: \n", returns)
    return df



def replay(fail_actions):
    """
    method runs through an action sequence and displays the actions the agent tries until it fails.

    :param fail_actions:  actions of a failed episode.
    """

    print("replay start")
    reset()

    for idx, action in enumerate(fail_actions):
        print("agent position: ", env.agent_pos)
        time.sleep(args.pause)
        
        obs, _, _, _ = env.step(action)
        redraw(obs)

        action_name =  action_to_string(action)
        print("step: {}, action: {}".format(idx, action_name))
    
    print("replay finished")  
    window.close()    
    return

  

def main(args):
    """
    method that loads the replay data and sets up the environment. 
    It initiates the preprocessing of the action sequence and finally starts the replay feature.
    """
    
    # load replay data
    replay_data = read_action_file(args.path)
    
    global seed
    seed = int(replay_data.iloc[:, :1].values[args.replay_no])
    print("seed: ", seed)

    global env
    env = gym.make(args.env)
    gym.Env.seed(env, seed)


    if args.agent_view:
       env = RGBImgPartialObsWrapper(env)
       env = ImgObsWrapper(env)

    global window
    window = Window('gym_minigrid - ' + args.env)
    #window.reg_key_handler(key_handler)

    reset()
    print("initial agent x_position: ", env.agent_pos[0])
    print("initial agent y_position: ", env.agent_pos[1])
    
    plt.ioff()
    # data preprocessing
    action_seq = []

    #print(replay_data.iloc[args.replay_no, 1], "type: ", type(replay_data.iloc[args.replay_no, 1]),  len(replay_data.iloc[args.replay_no, 1]))

    # from string to list of integers
    action_list = [int(x) for x in replay_data.iloc[args.replay_no, 1].lstrip('[').rstrip(']').split(',') if x.strip().isdigit()]              
    #print(action_list, type(action_list), len(action_list))

    for _, action_no in enumerate(action_list): 
        env_action = map_number_to_action(action_no)   
        action_seq.append(env_action)
        #print(action_seq)
    
    # show the actions that the agent tries
    replay(action_seq)

    # blocking event loop
    #window.show(block=True)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)






