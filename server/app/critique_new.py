import datetime

import dash
from dash import html, dcc, Input, Output, callback, no_update, State
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from dash.exceptions import PreventUpdate
from app.data_writer import write_user_data_interactions

# icons for buttons and feedback segment containers
plus_icon = html.I(className="bi bi-plus", style={'font-size': '20px'})
minus_icon = html.I(className="bi bi-dash", style={'font-size': '20px'})

# icons for critique buttons:
thumbs_up_icon = html.I(className="bi bi-hand-thumbs-up", style={'font-size': '20px', 'color': 'green'})
reset_icon = html.I(className="bi bi-x", style={'font-size': '20px', 'color': 'grey'})
thumbs_down_icon = html.I(className="bi bi-hand-thumbs-down",
                          style={'font-size': '20px', 'color': 'red'})


# subtract neighbouring slider values and return the list of differences; the differences correspond to the amounts
# of steps that belong to the step segments:
def diffs(slider_values):
    return [j - i for i, j in zip(slider_values[:-1], slider_values[1:])]


# old version of critique_layout_new when i was still using the feedback div:
# critique_layout_new = html.Div([
#     dcc.Store('step_scores'),
#     dcc.Store('step_colors'),
#     html.Div([
#         html.Div([
#             html.Div([
#                 html.Label('Feedback for steps:', style={'padding': '3px'}),
#                 html.Label('no steps selected', id='selected_steps_label_new',
#                            style={'padding': '3px', 'font-weight': 'bold'})
#             ], style={'display': 'flex'}),
#             html.Div([
#                 html.Div([
#                     html.Div([
#                         html.Div([
#                             dbc.Button([thumbs_up_icon], id='thumbs_up_btn_new', className='slider-button',
#                                        style={'left': '50%', 'position': 'relative',
#                                               'transform': 'translate(-50%, 0%)'}),
#                         ]),
#                         html.Label('good', style={'font-size': '1.5vh'})
#                     ], style={'padding-top': '0.5vh', 'margin-left': '1vw'}),
#                     html.Div([
#                         html.Div([
#                             dbc.Button([thumbs_down_icon], id='thumbs_down_btn_new', disabled=False,
#                                        className='slider-button',
#                                        style={'left': '50%', 'position': 'relative',
#                                               'transform': 'translate(-50%, 0%)'}),
#                         ]),
#                         html.Label('bad', style={'font-size': '1.5vh'})
#                     ], style={'padding-top': '0.5vh', 'margin-left': '1vw'}),
#                     html.Div([
#                         html.Div([
#                             dbc.Button([reset_icon], id='reset_btn_new', disabled=False, className='slider-button',
#                                        style={'left': '50%', 'position': 'relative',
#                                               'transform': 'translate(-50%, 0%)'}),
#                         ]),
#                         html.Label('neutral', style={'font-size': '1.5vh'})
#                     ], style={'padding-top': '0.5vh', 'margin-left': '1vw'}),
#                     # dbc.Button([thumbs_up_icon], id='thumbs_up_btn',
#                     #            style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
#                     #                   'margin-right': '0.5vw', 'margin-left': '2vw'}),
#                     # dbc.Button([thumbs_down_icon], id='thumbs_down_btn', disabled=False,
#                     #            style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
#                     #                   'margin-right': '0.5vw', }),
#                     # dbc.Button([reset_icon], id='reset_btn', disabled=False,
#                     #            style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
#                     #                   'margin-right': '0.5vw', })
#                 ], style={'width': '10vw', 'position': 'relative', 'margin': '0 auto', 'display': 'flex'}),
#             ])
#         ], style={'width': '26vw'}),
#     ], style={'display': 'flex', 'border-radius': '5px', 'border': '1px solid #bbb', 'width': '26vw',
#               'margin-top': '1vw',
#
#               'background-color': 'rgb(242,242,242)'}),
#     # button to submit the feedback:
#     dbc.Button('Submit', id='c_submit_button_new', style={'margin-top': '0.5vw'},
#                disabled=False)
# ])

critique_layout_new = html.Div([
    dcc.Store('step_scores'),
    dcc.Store('step_colors'),
    dcc.Store('step_scores_intermediate'),  # needed to submit feedback with autosubmit
    dcc.Store('last_labeled_episode'),
    html.Div([
        html.Div([
            html.Div([
                html.Label('Feedback for steps:', style={'padding': '3px', 'padding-left': '0.5vw'}),
                html.Label('No steps selected.', id='selected_steps_label_new',
                           style={'padding-left': '0.5vw', 'font-weight': 'bold', 'padding': '3px', 'width': '10vw'})
            ], style={'display': 'flex'}),
            html.Div([
                html.Div([
                    html.Div([
                        dbc.Button([thumbs_up_icon, 'good'], id='thumbs_up_btn_new', className='slider-button',
                                   style={'left': '0%', 'position': 'relative', 'transform': 'translate(0%, 0%)'})
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw', 'display': 'flex'}),
                    html.Div([
                        dbc.Button([thumbs_down_icon, 'bad'], id='thumbs_down_btn_new', disabled=False,
                                   className='slider-button',
                                   style={'left': '0%', 'position': 'relative', 'transform': 'translate(0%, 0%)'}),
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw', 'display': 'flex'}),
                    html.Div([
                        dbc.Button([reset_icon, 'neutral'], id='reset_btn_new', disabled=False,
                                   className='slider-button',
                                   style={'left': '0%', 'position': 'relative', 'transform': 'translate(0%, 0%)'}),
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw', 'display': 'flex'}),
                    html.Div([
                        dbc.Button([thumbs_up_icon, 'leftover'], id='leftover_good_btn', className='slider-button',
                                   style={'left': '0%', 'position': 'relative', 'transform': 'translate(0%, 0%)'})
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw', 'display': 'flex'}),
                    html.Div([
                        dbc.Button([thumbs_down_icon, 'leftover'], id='leftover_bad_btn', disabled=False,
                                   className='slider-button',
                                   style={'left': '0%', 'position': 'relative', 'transform': 'translate(0%, 0%)'}),
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw', 'display': 'flex'}),
                    # dbc.Button([thumbs_up_icon], id='thumbs_up_btn',
                    #            style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
                    #                   'margin-right': '0.5vw', 'margin-left': '2vw'}),
                    # dbc.Button([thumbs_down_icon], id='thumbs_down_btn', disabled=False,
                    #            style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
                    #                   'margin-right': '0.5vw', }),
                    # dbc.Button([reset_icon], id='reset_btn', disabled=False,
                    #            style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
                    #                   'margin-right': '0.5vw', })
                ], style={'width': '10vw', 'position': 'relative', 'margin': '0 auto', 'display': 'flex'}),
            ])
        ], style={'width': '62vw', 'display': 'flex'}),
    ], style={'display': 'flex', 'margin-left': '0.5vw', 'margin-top': '1.5vh', 'height': '3.5vh', 'width': '62vw',
              'background-color': '#EFF6FB'}),
])


@callback(Output('thumbs_up_btn_new', 'disabled'),
          Output('thumbs_down_btn_new', 'disabled'),
          Output('reset_btn_new', 'disabled'),
          Input('selected_steps', 'data'),
          config_prevent_initial_callbacks=True)
def set_critique_buttons_disabled(selected_steps):
    if selected_steps is not None:
        return False, False, False
    else:
        return True, True, True


@callback(Output('selected_steps_label_new', 'children'),
          Input('selected_steps', 'data'),
          config_prevent_initial_callbacks=True)
def update_selected_steps_label_new(steps):
    if steps is not None:
        # sort steps:
        steps = sorted(steps, key=lambda k: k['x'])
        # cut off steps if len(steps) > 5:
        if len(steps) > 5:
            steps = steps[:5]
            longer = True
        else:
            longer = False
        # concatenate steps to string:
        steps = ', '.join([str(step['x']) for step in steps])
        if longer:
            steps += ', ...'
        return steps
    else:
        return 'No steps selected.'


@callback(Output('c_submit_button_new', 'disabled'),
          Input('feedback_tabs', 'value'),
          Input('uncertainty_display_episode', 'data'),
          config_prevent_initial_callbacks=True)
def set_submit_button_disabled(feedback_type, episode):
    if feedback_type != 'tab-3':
        return True
    if episode is None:
        return True
    return False


# callback to update the step scores and colors:
@callback(Output('step_scores', 'data'),
          Input('thumbs_up_btn_new', 'n_clicks'),
          Input('thumbs_down_btn_new', 'n_clicks'),
          Input('reset_btn_new', 'n_clicks'),
          Input('uncertainty_display_episode', 'data'),
          Input('leftover_good_btn', 'n_clicks'),
          Input('leftover_bad_btn', 'n_clicks'),
          State('selected_steps', 'data'),
          State('step_scores', 'data'),
          State('actions', 'data'),
          State('session_interaction_data_filename', 'data'),
          State('episode_idx', 'data'),
          State('proc_idx', 'data'),
          State('buffer_idx', 'data'),
          State('episode_ids', 'data'),
          State('proc_ids', 'data'),
          State('buffer_ids', 'data'),
          config_prevent_initial_callbacks=True)
def update_step_scores(thumbs_up_btn, thumbs_down_btn, reset_btn, episode, leftover_good_click, leftover_bad_click,
                       selected_steps, step_scores, actions, filename, e_idx, p_idx, b_idx, e_ids, p_ids, b_ids):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # timestamp with for logging
    time = datetime.datetime.now().strftime("%H:%M:%S")

    if triggered_id == 'uncertainty_display_episode':
        if episode is not None:
            if filename:
                e_idx = e_ids[episode]
                p_idx = p_ids[episode]
                b_idx = b_ids[episode]
                write_user_data_interactions(
                    [time, 'change episode', 'click', str(e_idx), str(p_idx), 'None', str(b_idx), str(episode)],
                    filename)
            return [0] * len(actions[episode])
        else:
            return None

    if triggered_id == 'thumbs_up_btn_new':
        if filename:
            e_idx = e_ids[episode]
            p_idx = p_ids[episode]
            b_idx = b_ids[episode]
            details = [point['x'] for point in selected_steps]
            write_user_data_interactions(
                [time, 'thumbs_up_btn_new', 'click', str(e_idx), str(p_idx), 'see details', str(b_idx), str(details)],
                filename)
        for step in selected_steps:
            step_scores[step['x']] = 1
        return step_scores
    elif triggered_id == 'thumbs_down_btn_new':
        if filename:
            e_idx = e_ids[episode]
            p_idx = p_ids[episode]
            b_idx = b_ids[episode]
            details = [point['x'] for point in selected_steps]
            write_user_data_interactions(
                [time, 'thumbs_down_btn_new', 'click', str(e_idx), str(p_idx), 'see details', str(b_idx), str(details)],
                filename)
        for step in selected_steps:
            step_scores[step['x']] = -1
        return step_scores
    elif triggered_id == 'reset_btn_new':
        if filename:
            e_idx = e_ids[episode]
            p_idx = p_ids[episode]
            b_idx = b_ids[episode]
            details = [point['x'] for point in selected_steps]
            write_user_data_interactions(
                [time, 'reset_btn_new', 'click', str(e_idx), str(p_idx), 'see details', str(b_idx), str(details)],
                filename)
        for step in selected_steps:
            step_scores[step['x']] = 0
        return step_scores
    elif triggered_id == 'leftover_good_btn':
        if filename:
            e_idx = e_ids[episode]
            p_idx = p_ids[episode]
            b_idx = b_ids[episode]
            write_user_data_interactions(
                [time, 'leftover_good_btn', 'click', str(e_idx), str(p_idx), 'None', str(b_idx), 'None'], filename)
        for i, score in enumerate(step_scores):
            if score == 0:
                step_scores[i] = 1
        return step_scores
    else:
        if filename:
            e_idx = e_ids[episode]
            p_idx = p_ids[episode]
            b_idx = b_ids[episode]
            write_user_data_interactions(
                [time, 'leftover_bad_btn', 'click', str(e_idx), str(p_idx), 'None', str(b_idx), 'None'], filename)
        for i, score in enumerate(step_scores):
            if score == 0:
                step_scores[i] = -1
        return step_scores


# callback to update step_scores_intermediate:
@callback(Output('step_scores_intermediate', 'data'),
          Input('step_scores', 'data'),
          State('session_data_filename', 'data'),
          config_prevent_initial_callbacks=True)
def update_step_scores_intermediate(step_scores, session_datafile_path):
    if session_datafile_path is not None:
        return step_scores
    return no_update


# callback to update last_labeled_episode:
@callback(Output('last_labeled_episode', 'data'),
          Input('thumbs_up_btn_new', 'n_clicks'),
          Input('thumbs_down_btn_new', 'n_clicks'),
          Input('leftover_good_btn', 'n_clicks'),
          Input('leftover_bad_btn', 'n_clicks'),
          State('uncertainty_display_episode', 'data'),
          State('episode_ids', 'data'),
          State('proc_ids', 'data'),
          State('buffer_ids', 'data'),
          State('rewards_episode_wise', 'data'),
          State('loss_episode_wise', 'data'),
          State('entropy_stepwise', 'data'),
          config_prevent_initial_callbacks=True)
def update_last_labeled_episode(thumbs_up_btn, thumbs_down_btn, leftover_good, leftover_bad, episode, episode_ids,
                                proc_ids, buffer_ids, rewards_episode_wise,
                                loss_episode_wise, entropy_stepwise):
    return {'episode': episode,
            'episode_idx': episode_ids[episode],
            'proc_idx': proc_ids[episode],
            'buffer_idx': buffer_ids[episode],
            'mean_reward': rewards_episode_wise[episode],
            'mean_loss': loss_episode_wise[episode],
            'entropy': entropy_stepwise[episode]}
