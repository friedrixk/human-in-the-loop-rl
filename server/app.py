import multiprocessing
import os
from multiprocessing import freeze_support

import dash
from dash import Dash, html, dcc, Input, Output, State, no_update
from app.loss_reward import loss_reward
from app.uncertainty import uncertainty
from app.feedback import feedback
from app.train_rl import train_rl
import psutil
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import pickle
import time
from flask import send_file
import shutil
from app.data_writer import generate_feedback_file, generate_interactions_file, write_amount_feedback_frames, \
    write_user_data_interactions
import datetime

try:
    with open('initial_data.pickle', 'rb') as file:
        initial_data = pickle.load(file)
    print("== INITIAL DATA FOUND AND LOADED")
except FileNotFoundError:
    initial_data = pd.DataFrame()
    print("== INITIAL DATA NOT FOUND")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.icons.BOOTSTRAP]

# icons for menu buttons
play_icon = html.I(className="bi bi-play", style={'font-size': '2.5vh', 'color': '#00994C'})
pause_icon = html.I(className="bi bi-pause", style={'font-size': '2.5vh'})
save_icon = html.I(className="bi bi-save", style={'font-size': '2.5vh'})
stop_icon = html.I(className="bi bi-stop", style={'font-size': '2.5vh', 'color': '#CC0000'})
refresh_icon = html.I(className="bi bi-arrow-clockwise", style={'font-size': '2vh'})
download_icon = html.I(className="bi bi-download", style={'font-size': '2vh'})
time_elapsed = '00:00:00'

# list of all possible environments
env_list = ['existing model', 'GoToObj', 'GoToRedBallGrey', 'GoToRedBall', 'GoToLocal', 'PutNextLocal', 'PickupLoc',
            'GoToObjMaze',
            'GoTo', 'Pickup', 'UnblockPickup', 'Open', 'Unlock', 'PutNext', 'Synth', 'SynthLoc', 'GoToSeq', 'SynthSeq',
            'GoToImpUnlock', 'BossLevel', 'GoToRedSwitchGrey', 'GoToRedSwitch', 'Flip', 'FlipDist', 'FlipBig',
            'GoToOtherRoomOpen', 'GoToOtherRoomClosed', 'GoToOtherRoomLocked', 'FlipInOtherRoomSwitch', 'FlipRoom',
            'PickUpOtherRoom', 'UncoverUnspecifiedDoor', 'GoToUnspecifiedRoom', 'BlockedRoomBall', 'OpenMoveBox',
            'BlockedRoomMovBox', 'BlockedRoomUnspecified', 'UnlockSwitch']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# layout
app.layout = html.Div([
    dcc.Store(id='amount_feedback_frames'),
    dcc.Store(id='num_episodes'),
    dcc.Store(id='session_data_filename'),
    dcc.Store(id='session_interaction_data_filename'),
    dcc.Store(id='proxy'),
    dcc.Store(id='dictionary', data=initial_data),  # when providing inital data: data=initial_data
    dcc.Store(id='training_pid'),
    dcc.Interval(id='refresh_data_interval', interval=15000),
    html.Div(id='time_counter_interval_div'),
    html.H4('Human in the Loop RL - Workbench', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            # div for the download links:
            html.Div([
                html.A('download models', id='models_download_link', download='models.zip', href='/download/models.zip',
                       style={'margin-right': '3px'}),
                html.A('download user data', id='user_data_download_link', download='user_data.zip',
                       href='/download/user_data.zip', style={'margin-right': '3px'}),
            ], style={'display': 'flex'}),
            # Dropdown to select environment
            dcc.Dropdown(env_list, placeholder='Select env', id='env_dropdown',
                         style={'margin-right': '3px', 'width': '10vw'}),
            # Dropdown to select a model
            dcc.Dropdown(id='model_dropdown', placeholder='Select model', style={'width': '15vw'},
                         optionHeight=100),
            dbc.Button([play_icon], id='start_button',
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'margin-left': '3px',
                              'height': '36px'}),
            dbc.Tooltip('start training', target='start_button', placement='top'),
            dbc.Button([stop_icon], id='stop_button', disabled=False,
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'margin-left': '3px',
                              'height': '36px'}),
            dbc.Tooltip('save model and stop training', target='stop_button', placement='top'),
            dbc.Button(dbc.Spinner(id='save_spinner', size='sm'), id='save_button', disabled=True,
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'margin-left': '3px',
                              'height': '36px'}),
            dbc.Tooltip('save model', target='save_button', placement='top'),
            dbc.Button([pause_icon], id='pause_button',
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'margin-left': '3px',
                              'height': '36px'}),
            dbc.Tooltip('save current data', target='pause_button', placement='top'),
            dbc.Button([refresh_icon], id='refresh_button', disabled='True',
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
                              'margin-left': '3px'}),
            dbc.Tooltip('display new data', target='refresh_button', placement='top'),
            html.Div(id='notification_dot',
                     style={'height': '10px', 'width': '10px', 'border-radius': '5px', 'margin-left': '-5px',
                            'margin-top': '-3px'})
        ], style={'margin-left': '2vw', 'display': 'flex'}),
        html.Div([
            html.Label(id='time_counter', children='Time elapsed: ' + time_elapsed,
                       style={'margin': 0, 'position': 'relative', 'top': '50%', '-ms-transform': 'translateY(-50%)',
                              'transform': 'translateY(-50%'})
        ], style={'margin-left': '2vw'}),
        html.Div([
            html.Label('Training progress: ',
                       style={'margin': 0, 'position': 'relative', 'top': '50%',
                              '-ms-transform': 'translateY(-50%)',
                              'transform': 'translateY(-50%)'})
        ], style={'margin-left': '2vw'}),
        html.Div([
            dbc.Progress(id='progress_bar', value=0, striped=True, label='0%',
                         style={'font-size': '1rem', 'height': '0.8vw', 'width': '20vw', 'margin': 0,
                                'position': 'relative', 'top': '50%', '-ms-transform': 'translateY(-50%)',
                                'transform': 'translateY(-50%)'})
        ], style={'margin-left': '0.5vw'})
    ], style={'display': 'flex'}),
    html.Div(children=[
        html.Div([

            # overview-div
            html.Div([
                # div that contains all information on loss and reward
                html.Div([
                    loss_reward,
                ], id='gradient_norm-div',
                    #style={'padding-right': '1vw', 'padding-left': '1vw', 'padding-top': '1vh', 'padding-bottom': '1vh',
                           #'background-color': '#f4f4f4'}
),
            ], id='overview-div'),

            # div that contains all information on uncertainty
            html.Div([
                html.Div([
                    uncertainty,
                ], style={'padding-right': '1vw', 'padding-left': '1vw', 'padding-bottom': '1vh',
                          'background-color': '#f4f4f4'})
            ], id='uncertainty-div')
        ]),

        # div that contains all functionalities for human based feedback
        html.Div([
            html.Div([
                feedback
            ], style={'padding-right': '1vw', 'padding-left': '1vw', 'padding-bottom': '1vh',
                      'background-color': '#f9f9f9', 'width': '28vw', 'height': '85vh', 'display': 'none'}),
        ], id='feedback-div')

    ], style={'display': 'flex'}, className='row')

])


# @app.callback(
#     Output(component_id='feedback-div', component_property='style'),
#     Input(component_id='checklist', component_property='value')
# )
# def update_hide_status_feedback_div(checklist):
#     if 'HITL feedback' in checklist:
#         width = '28vw' if len(checklist) > 1 else '100vw'
#         return {'display': 'flex', 'text-align': 'center',
#                 'height': '85vh', 'width': width}
#     else:
#         return {'display': 'none'}


# @app.callback(
#     Output(component_id='overview-div', component_property='style'),
#     Input(component_id='checklist', component_property='value')
# )
# def update_hide_status_overview_div(checklist):
#     if 'Gradient Norm' in checklist:
#         width = '70vw' if 'HITL feedback' in checklist else '100vw'
#         height = '44vh' if 'Uncertainty' in checklist else '88vh'
#         return {'display': 'flex', '''background-color': '#99CCFF',''' 'height': height, 'width': width}
#     else:
#         return {'display': 'none'}


# @app.callback(
#     Output(component_id='uncertainty-div', component_property='style'),
#     Input(component_id='checklist', component_property='value')
# )
# def update_hide_status_uncertainty_div(checklist):
#     if 'Uncertainty' in checklist:
#         width = '70vw' if 'HITL feedback' in checklist else '100vw'
#         height = '44vh' if 'Gradient Norm' in checklist else '88vh'
#         return {'display': 'flex', ''''background-color': '#FF9999',''' 'height': height, 'width': width}
#     else:
#         return {'display': 'none'}


@app.callback(
    Output('start_button', 'disabled'),
    Output('stop_button', 'disabled'),
    Output('env_dropdown', 'disabled'),
    Output('progress_bar', 'animated'),
    Output('time_counter_interval_div', 'children'),
    Output('training_pid', 'data'),
    Input('start_button', 'n_clicks'),
    Input('stop_button', 'n_clicks'),
    State('env_dropdown', 'value'),
    State('model_dropdown', 'value'),
    State('training_pid', 'data'),
    State('amount_feedback_frames', 'data'),
    State('num_episodes', 'data'),
    State('session_data_filename', 'data')
)
def start_stop_training(start_clicks, stop_clicks, env, model, pid, amount_feedback_frames, num_episodes,
                        session_data_filename):
    # get id of component that triggered the callback
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # start the training:
    if trigger_id == 'start_button':
        if env:
            if env == 'existing model' and model is None:
                return no_update
            p2 = multiprocessing.Process(target=train_rl, args=('BabyAI-' + env + '-v0', model, q0, q1, q2))
            p2.start()
            # start counting elapsed time
            interval = dcc.Interval(id='time_counter_interval', interval=1000)
            return True, False, True, True, interval, p2.pid
        else:
            return no_update

    # stop the training:
    if trigger_id == 'stop_button':
        # tell train_rl.py to save the model:
        q0.put('save')
        while q0.qsize() > 0:
            time.sleep(1)
        write_amount_feedback_frames([amount_feedback_frames, num_episodes], session_data_filename)
        # zip the model folder:
        shutil.make_archive('models', 'zip', './models')
        # zip the user_data folder:
        shutil.make_archive('user_data', 'zip', './user_data')
        # killing all training processes:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
        # emptying the queues
        while not q0 \
                .empty():
            q0.get()
        while not q1.empty():
            q1.get()
        while not q2.empty():
            q2.get()
        print('Training stopped.')
        return False, True, False, False, None, None
    else:
        return no_update


@app.callback(
    Output('dictionary', 'data'),
    Output('progress_bar', 'value'),
    Output('progress_bar', 'label'),
    Output('notification_dot', 'style'),
    Output('amount_feedback_frames', 'data'),
    Output('num_episodes', 'data'),
    Input('refresh_data_interval', 'n_intervals'),
    Input('refresh_button', 'n_clicks'),
    Input('start_button', 'n_clicks')
)
def refresh_data(n_intervals, refresh_click, start_click):
    """
    Collects new randomly assembled training data that is procured via the queue and stores (not visualizes!) it in the
    dcc.Store component with id='dictionary' (see layout). It also updates the style of the notification dot that appears
    when in the top right corner of the refresh button when new training data is available in the queue. Also, the
    progress bar value and label are updated with the latest values coming from the queue.
    TODO: replace this method with a manual request for data so that data only gets updated when requested by the human
    @param n_intervals: the number of times the queue has been checked for new training data
    @param refresh_click: the number of times the refresh button has been clicked
    @param start_click: the number of times the start button has been clicked
    @return:
    """
    # get id of component that triggered the callback
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'start_button':
        # reset style of the notification dot
        style = {'height': '10px', 'width': '10px', 'border-radius': '5px', 'margin-left': '-5px', 'margin-top': '-3px'}
        return no_update, None, None, style, None, 0

    if trigger_id == 'refresh_button':
        # reset style of the notification dot
        style = {'height': '10px', 'width': '10px', 'border-radius': '5px', 'margin-left': '-5px', 'margin-top': '-3px'}
        return no_update, no_update, no_update, style, no_update, no_update

    print('Try getting data from queue.')
    if not q1.empty():
        print('q1 is not empty')
        data = q1.get()
        value = round(data[1], 10)
        label = f'{round(value, 2)}%'
        print(f'progress: {label}')
        style = {'height': '10px', 'width': '10px', 'border-radius': '5px',
                 'background-color': '#7cee70', 'margin-left': '-5px', 'margin-top': '-3px'}
        amount_feedback_frames = data[2]
        print(
            f'amount_feedback_frames: {amount_feedback_frames}')
        num_episodes = data[3]
        print(f'frontend num_episodes: {num_episodes}')
        return data[0], value, label, style, amount_feedback_frames, num_episodes
    return no_update


@app.callback(
    Output('time_counter', 'children'),
    Input('time_counter_interval', 'n_intervals'),
)
def update_time_elapsed(n_intervals):
    if n_intervals:
        hours = int(n_intervals / 3600)
        minutes = int(n_intervals / 60) % 60
        seconds = int(n_intervals) % 60
        h = f'0{hours}' if hours < 10 else f'{hours}'
        m = f'0{minutes}' if minutes < 10 else f'{minutes}'
        s = f'0{seconds}' if seconds < 10 else f'{seconds}'
        elapsed_time = f'Time elapsed: {h}:{m}:{s}'
        return elapsed_time
    else:
        return no_update


@app.callback(
    Output('agent_x_pos', 'data'),
    Output('agent_y_pos', 'data'),
    Output('agent_dir', 'data'),
    Output('grid_width', 'data'),
    Output('grid_height', 'data'),
    Output('loss_episode_wise', 'data'),
    Output('rewards_episode_wise', 'data'),
    Output('entropy_stepwise', 'data'),
    Output('entropy_episode_wise', 'data'),
    Output('episodes_fail', 'data'),
    Output('episodes_success', 'data'),
    Output('feedback_episodes', 'data'),
    Output('imgs', 'data'),
    Output('actions', 'data'),
    Output('missions', 'data'),
    Output('proc_ids', 'data'),
    Output('episode_ids', 'data'),
    Output('buffer_ids', 'data'),
    Input('refresh_button', 'n_clicks'),
    Input('last_labeled_episode', 'data'),
    State('feedback_episodes', 'data'),
    State('dictionary', 'data'),
    State('session_interaction_data_filename', 'data'),
    State('session_data_filename', 'data'),
)
def refresh_vis_data(n_clicks, last_labeled_episode, feedback_episodes, vis_dict, interaction_file,
                     feedback_file):
    """
    Actually visualize the data that has already been retrieved and stored through refresh_data().
    @param n_clicks: the number of times the refresh button has been clicked
    @param vis_dict: the data to visualize
    @return:
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    # when feedback has been given (i.e. feedback_scores has changed), update feedback_episodes:
    if trigger_id == 'last_labeled_episode':
        feedback_episodes[last_labeled_episode['episode']] = True
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, \
               no_update, no_update, feedback_episodes, no_update, no_update, no_update, no_update, no_update, no_update

    if trigger_id == 'refresh_button':
        # timestamp with for logging
        time = datetime.datetime.now().strftime("%H:%M:%S")
        # write interaction in log file:
        write_user_data_interactions(
            [time, 'refresh button', 'click', 'None', 'None', 'None', 'unknown', 'None'], interaction_file)
        write_amount_feedback_frames([time, 'refresh button clicked', 'unknown'], feedback_file)

    if not vis_dict:
        return no_update
    else:
        # setting the dcc.Store variables in loss_reward.py with new data:
        # loss_stepwise = list(vis_dict['loss_stepwise'].values())
        # policy_loss_stepwise = list(vis_dict['policy_loss_stepwise'].values())
        # value_loss_stepwise = list(vis_dict['value_loss_stepwise'].values())
        # print(f'policy_loss_stepwise: {policy_loss_stepwise}')
        # print(f'value_loss_stepwise: {value_loss_stepwise}')
        # print(f'loss_stepwise: {loss_stepwise}')
        # loss_episode_wise = [round(sum(e) / len(e), 2) for e in loss_stepwise]
        loss_episode_wise = list(vis_dict['loss_episode_wise'].values())
        # rewards_stepwise = list(vis_dict['rewards'].values())
        # divide by 20 (the reward-scale, see train_rl.py) and 2 (maximum possible reward with feedback) to normalize
        # the rewards to the range [0, 1]:
        # rewards_episode_wise = [round(np.mean(e) / 20 / 2, 2) for e in rewards_stepwise]
        rewards_episode_wise = list(vis_dict['reward_episode_wise'].values())
        episodes_fail = [ind for ind, e in enumerate(rewards_episode_wise) if e == 0]
        episodes_success = [ind for ind, e in enumerate(rewards_episode_wise) if e > 0]
        # get the information what episodes have already received feedback and flatten the list:
        feedback_episodes = [e[:1] for e in list(vis_dict['feedback'].values())]
        feedback_episodes = [item for sublist in feedback_episodes for item in sublist]

        # setting the dcc.Store variables in uncertainty.py (for performance reasons in grid graph use numpy arrays):
        entropy_max = - ((1 / 7) * np.log(1 / 7)) * 7
        entropy_stepwise = np.array(
            [np.round(np.array(ls) / entropy_max, 2) for ls in (list(vis_dict['entropy_stepwise'].values()))],
            dtype=object)
        entropy_episode_wise = [round(sum(e) / len(e), 2) for e in entropy_stepwise]
        agent_x_pos = np.array([np.array(ls) for ls in (list(vis_dict['agent_x_pos'].values()))], dtype=object)
        agent_y_pos = np.array([np.array(ls) for ls in (list(vis_dict['agent_y_pos'].values()))], dtype=object)
        agent_dir = np.array([np.array(ls) for ls in (list(vis_dict['agent_dir'].values()))], dtype=object)
        grid_width = np.array([np.array(ls) for ls in (list(vis_dict['grid_width'].values()))], dtype=object)
        grid_height = np.array([np.array(ls) for ls in (list(vis_dict['grid_height'].values()))], dtype=object)
        actions = np.array([np.array(ls) for ls in (list(vis_dict['actions'].values()))], dtype=object)
        missions = np.array(list(vis_dict['mission'].values()), dtype=object)
        # imgs = np.array([np.array(ls) for ls in (list(vis_dict['imgs'].values()))], dtype=object)
        imgs = np.array([np.array(ls) for ls in (list(vis_dict['human_views'].values()))], dtype=object)

        # setting the dcc.Store variables in feedback.py:
        proc_ids = np.array((list(vis_dict['proc_idx'].values())), dtype=int)
        episode_ids = np.array((list(vis_dict['episode_idx'].values())), dtype=int)
        buffer_ids = np.array((list(vis_dict['buffer_idx'].values())), dtype=int)
        print(buffer_ids)

        return agent_x_pos, agent_y_pos, agent_dir, grid_width, grid_height, loss_episode_wise, rewards_episode_wise, \
               entropy_stepwise, entropy_episode_wise, episodes_fail, episodes_success, feedback_episodes, imgs, actions, missions, proc_ids, episode_ids, buffer_ids


@app.callback(Output('proxy', 'data'),
              Input('feedback_scores', 'data'),
              Input('proc_idx', 'data'),
              Input('episode_idx', 'data'),
              Input('buffer_idx', 'data'),
              )
def put_feedback_in_queue(feedback, p, e, b):
    if feedback:
        print('putting feedback in queue')
        q2.put((feedback, p, e, b))
    return no_update


@app.callback(Output('refresh_button', 'disabled'),
              Input('dictionary', 'data'))
def set_refresh_button_disabled(data):
    if data is None:
        return True
    else:
        return False


@app.callback(
    Output('model_dropdown', 'disabled'),
    Output('model_dropdown', 'value'),
    Input('env_dropdown', 'value'))
def model_dropdown_disable(value):
    if value == 'existing model':
        return False, no_update
    else:
        return True, None


@app.callback(
    Output('model_dropdown', 'options'),
    Input('env_dropdown', 'value'))
def update_existing_models(env):
    # check if folder 'models' exists:
    if not os.path.exists('models'):
        # if not, return empty list:
        return []
    # if yes, check if there are any models in the folder:
    else:
        # if yes, return the list of the names of the models:
        if os.listdir('models'):
            print(os.listdir('models'))
            return os.listdir('models')
        # if not, return empty list
        else:
            return []


@app.callback(Output('pause_button', 'n_clicks'),
              Input('pause_button', 'n_clicks'),
              State('dictionary', 'data'))
def save_data(click, dictionary):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'pause_button':
        with open('initial_data.pickle', 'wb') as file:
            pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
    return no_update


# callback for the display of a spinner while the model is being saved:
@app.callback(Output('save_spinner', 'children'),
              Input('save_button', 'n_clicks'),
              State('amount_feedback_frames', 'data'),
              State('num_episodes', 'data'),
              State('session_data_filename', 'data'))
def save_spinner(click, amount_feedback_frames, num_episodes, session_data_filename):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'save_button':
        # tell train_rl.py to save the model:0
        q0.put('save')
        # while q0 is not empty, wait:
        while q0.qsize() > 0:
            time.sleep(1)
        write_amount_feedback_frames([amount_feedback_frames, num_episodes], session_data_filename)
        # zip the models folder:
        shutil.make_archive('models', 'zip', './models')
        # zip the user_data folder:
        shutil.make_archive('user_data', 'zip', './user_data')
        return save_icon
    else:
        return save_icon


# callback for dis- and enabling the save_button:
@app.callback(Output('save_button', 'disabled'),
              Input('start_button', 'n_clicks'),
              Input('stop_button', 'n_clicks'))
def set_save_button_disabled(start_click, stop_click):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'start_button':
        return False
    else:
        return True


# # Callback to handle download button click
# @app.callback(
#     Output('download_link', 'href'),
#     Input('download_button', 'n_clicks'))
# def handle_download(n_clicks):
#     shutil.make_archive('models', 'zip', './models')
#     return no_update


@app.server.route('/download/models.zip')
def download_models():
    return send_file('./models.zip')


@app.server.route('/download/user_data.zip')
def download_user_data():
    return send_file('./user_data.zip')


@app.callback(Output('session_data_filename', 'data'),
              Output('session_interaction_data_filename', 'data'),
              Input('start_button', 'n_clicks'),
              Input('stop_button', 'n_clicks'),
              State('model_dropdown', 'value'),
              State('env_dropdown', 'value'))
def set_sessions_data_path(start_click, stop_click, model, env):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'start_button':
        filename1 = generate_feedback_file()
        filename2 = generate_interactions_file()
        return filename1, filename2
    else:
        return None, None


if __name__ == '__main__':
    freeze_support()
    q0 = multiprocessing.Manager().Queue()
    q1 = multiprocessing.Manager().Queue()
    q2 = multiprocessing.Manager().Queue()
    app.run_server(debug=True, host='0.0.0.0')
