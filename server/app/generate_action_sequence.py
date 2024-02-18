
#!/usr/bin/env python3

import argparse
import gym
from gym_minigrid.window import Window

import matplotlib.pyplot as plt
from IPython import display





class ActionSequenceGenerator():

    # constructor
    def __init__(self, env_name, seed, agent_view, action_seq):
        self.env = gym.make(env_name)
        self.seed = seed
        self.agent_view = agent_view
        self.action_seq = action_seq


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
        

    def redraw(self, img):
      """
      method to update the window.

      :param img: new image to display
      """
      if not args.agent_view:
         img = self.env.render('rgb_array', tile_size=args.tile_size)

      window.show_img(img)   

    def redraw2(self, img):
     """
      method that updates the window
     """
     if not self.agent_view:
        img = self.env.render('rgb_array', tile_size=32)


     if self.type_of_script() == 'jupyter':
        if window.imshow_obj is None:
             window.imshow_obj = window.ax.imshow(img, interpolation='bilinear')

        window.imshow_obj.set_data(img)
        display.clear_output(wait=True) 
        display.display(plt.gcf())
     else:
        window.show_img(img)


    
    def reset(self):
       """
       method to reset the environment. It prints the mission and show the mission in the window.
       """
       self.action_seq.clear()

       if self.seed != -1:
           self.env.seed(self.seed)

       obs = self.env.reset()

       if hasattr(self.env, 'mission'):
           print('Mission: %s' % self.env.mission)
           window.set_caption(self.env.mission)

       self.redraw(obs)


     
   
    def step(self, action):
       """
       method that takes as input an action and invokes the environment's step method.
       If the mission is fulfilled after the given action, the method returns the sequence of actions that the human has entered. 
       If the mission is not fulfilled after the given action, the method invokes the redraw method

       :param action: agent's next action. 
       """
       obs, reward, done, _ = self.env.step(action)
       print('step=%s, reward=%.2f' % (self.env.step_count, reward))

       if done:
          print('end')
          window.close()
          print("action sequence: ", self.action_seq)
          return self.action_seq
          #self.reset()
       else:
          self.redraw(obs)


    
    def undo_step(self):
       """
       method that enables the human to undo a step/ an action.
       It removes the action from the action sequence that is used for the demo generation. 
       It resets the environment to the state before the action. 
       """

       print("undo step", self.env.step_count)
       print("sequence of actions before undoing last action: ", self.action_seq)

       if not self.action_seq:
          print("executing 'undo' is not possible because no action has been taken before")
       else:
          self.action_seq.pop()
          print("sequence of actions after undoing last action: ", self.action_seq)
       
       #print("agent position: ", self.env.agent_pos)
       
       self.env.seed(self.seed)
       obs = self.env.reset()

       for i in range(0, len(self.action_seq)):
          #print("iteration ", i)
          obs, _, _, _ = self.env.step(self.action_seq[i])

       self.redraw(obs)
       return


    
    def key_handler(self, event):
       """
       method handles key inputs. 
       If the key indicates an action, the method stores the action in a list that is used for the demo generation and it invokes the step method.
       If the user presses the undo key, the method invokes the undo_step method.

       :param event: click event
       """
       print('pressed', event.key)

       if event.key == 'escape':
          window.close()
          return

       if event.key == 'backspace':
          self.reset()
          return

       if event.key == 'left':
          self.action_seq.append(self.env.actions.left)
          self.step(self.env.actions.left)
          return

       if event.key == 'right':
          self.action_seq.append(self.env.actions.right)
          self.step(self.env.actions.right)
          return

       if event.key == 'up':
          self.action_seq.append(self.env.actions.forward)
          self.step(self.env.actions.forward)
          return

       if event.key == 'u':
          self.undo_step()
          return

    # Spacebar 
       if event.key == ' ':
          self.action_seq.append(self.env.actions.toggle)
          self.step(self.env.actions.toggle)
          return
       if event.key == 'pageup' or event.key == 'p':
          self.action_seq.append(self.env.actions.pickup)
          self.step(self.env.actions.pickup)
          return
       if event.key == 'pagedown' or event.key == 'd':
          self.action_seq.append(self.env.actions.drop)
          self.step(self.env.actions.drop)
          return

       if event.key == 'enter':       
          self.reset()
        

     
    def start_generation(self, env_name):
       """
       method that prepares the environment and the window to enable the human to enter the actions

       :param env_name: level name, e.g., BabyAI-GoTo-v0 
       """
     
       if self.agent_view:
          env = RGBImgPartialObsWrapper(self.env)
          env = ImgObsWrapper(env)

       global window
       window = Window('gym_minigrid - ' + env_name)
       window.reg_key_handler(self.key_handler)

       self.reset()

       # blocking event loop
       #window.show(block=True)
    


    
    

## for manual testing

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='BabyAI-BossLevel-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
       default=False,
      help="draw the agent sees (partially observable view)",
    action='store_true'
)


def main(args):

   actions = []

   action_seq_generator = ActionSequenceGenerator(args.env, args.seed, args.agent_view, actions)
   action_seq_generator.start_generation(args.env)

   # blocking event loop
   window.show(block=True)
   

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)