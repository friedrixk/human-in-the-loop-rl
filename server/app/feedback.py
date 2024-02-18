import dash
from dash import html, dcc, Input, Output, callback, no_update, State, ALL
import numpy as np
import math
from app.critique import critique_layout, diffs
from app.critique_new import critique_layout_new
from app.natural_language import natural_language_feedback_layout, sentiment_analysis
from app.data_writer import write_user_data_feedback
import datetime
import json

# layout:
feedback = html.Div([
    dcc.Store(id='segment_button_workaround', data=0),
    dcc.Store(id='proc_ids'),
    dcc.Store(id='episode_ids'),
    dcc.Store(id='buffer_ids'),
    dcc.Store(id='feedback_scores'),
    dcc.Store(id='proc_idx'),
    dcc.Store(id='episode_idx'),
    dcc.Store(id='buffer_idx'),
    dcc.Store(id='selected_step_sequence'),
    dcc.Store(id='critique_segment_colors', data=['#c6c3c2']),
    dcc.Store(id='slider_values'),
    dcc.Store(id='slider_max'),
    dcc.Store(id='critique_graph_hoverData'),
    dcc.ConfirmDialog(
        id='confirm_submit',
        message='Your feedback has been submitted successfully.',
    ),
    html.Div([
        html.Div([
            dcc.Tabs([
                dcc.Tab(label='critique', children=critique_layout, className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='natural language', children=natural_language_feedback_layout, className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='critique_new', className='custom-tab',  # children=critique_layout_new
                        selected_className='custom-tab--selected'),
            ], id='feedback_tabs', value='tab-3',
                style={'width': '26vw', 'height': '30px', 'padding': '0px', 'margin-bottom': '1vw'}),
        ]),
    ], style={'display': 'flex', 'margin-bottom': '5px', 'margin-top': '0.5vw'})
])


@callback(Output('confirm_submit', 'displayed'),
          Input('nl_submit_button', 'n_clicks'),
          Input('c_submit_button', 'n_clicks'),
          Input('c_submit_button_new', 'n_clicks'),
          config_prevent_initial_callbacks=True)
def display_confirm_submit(nl_submit_click, c_submit_click, c_submit_click_new):
    return True


# # this callback is used to update the children property of the div with id 'feedback_type_container' based on the value
# # of the dropdown with id 'feedback_type_dropdown':
# @callback(Output('feedback_type_container', 'children'),
#           Input('feedback_type_dropdown', 'value'),
#           config_prevent_initial_callbacks=True)
# def update_feedback_type_container(feedback_type):
#     if feedback_type == 'critique':
#         return critique_layout
#     if feedback_type == 'natural language':
#         return natural_language_feedback_layout
#     else:
#         return None


@callback(Output('slider_values', 'data'),
          Input('uncertainty_display_episode', 'data'),
          Input('c_slider', 'value'),
          Input('nl_slider', 'value'),
          Input('feedback_tabs', 'value'),
          State('segmentation_mode_dropdown', 'value'),
          State('actions', 'data'),
          Input({'type': 'feedback_segment', 'index': ALL}, 'value'),
          config_prevent_initial_callbacks=True)
def update_slider_values_store(episode, c_slider_values, nl_slider_values, feedback_tab, segmentation_mode, actions,
                               feedback_segments):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    # set slider values according to segments when nl segmentation mode is 'simple' (needed for uncertainty bg colors):
    if trigger_id not in ['uncertainty_display_episode', 'c_slider', 'nl_slider', 'feedback_tabs']:
        if segmentation_mode == 'simple':
            if len(feedback_segments) != 0:
                steps_per_segment = len(actions[episode]) // len(feedback_segments)
                n = steps_per_segment
                slider_values = []
                while n < len(actions[episode]):
                    slider_values.append(n)
                    n += steps_per_segment
                return slider_values
            else:
                return no_update
        else:
            return no_update

    # reset slider values if the episode changes or if the feedback tab changes:
    if trigger_id == 'uncertainty_display_episode' or trigger_id == 'feedback_tabs':
        return []
    # update slider values based on natural language slider:
    elif trigger_id == 'nl_slider' and feedback_tab == 'tab-2':
        return nl_slider_values
    # update slider values based on critique slider:
    else:
        return c_slider_values


@callback(Output('slider_max', 'data'),
          Input('c_slider', 'max'),
          Input('nl_slider', 'max'),
          State('feedback_tabs', 'value'),
          config_prevent_initial_callbacks=True)
def update_slider_max_store(c_slider_max, nl_slider_max, feedback_type):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'c_slider' and feedback_type == 'tab-1':
        return c_slider_max
    else:
        return nl_slider_max


@callback(Output('feedback_scores', 'data'),
          Output('proc_idx', 'data'),
          Output('episode_idx', 'data'),
          Output('buffer_idx', 'data'),
          Input('c_submit_button', 'n_clicks'),
          Input('nl_submit_button', 'n_clicks'),
          # Input('c_submit_button_new', 'n_clicks'),
          State('segmentation_mode_dropdown', 'value'),
          State({'type': 'feedback_segment', 'index': ALL}, 'value'),
          Input('uncertainty_display_episode', 'data'),
          State('actions', 'data'),
          State('proc_ids', 'data'),
          State('episode_ids', 'data'),
          State('buffer_ids', 'data'),
          State('nl_slider', 'value'),
          State('nl_slider', 'min'),
          State('nl_slider', 'max'),
          State('critique_segment_colors', 'data'),
          State('c_slider', 'value'),
          State('c_slider', 'max'),
          State('session_data_filename', 'data'),
          State('step_scores', 'data'),
          State('step_scores_intermediate', 'data'),
          State('last_labeled_episode', 'data'),
          config_prevent_initial_callbacks=True
          )
def submit_feedback(c_submit_clicks, nl_submit_clicks, segmentation_mode, feedback_segments,
                    episode, actions, proc_ids,
                    episode_ids, buffer_ids, slider_values, slider_min, slider_max, colors, c_slider_values,
                    c_slider_max,
                    session_data_filename, step_scores, step_scores_intermediate, last_labeled_episode):
    # timestamp with for logging
    time = datetime.datetime.now().strftime("%H:%M:%S")

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # if the episode has changed:
    if trigger_id == 'uncertainty_display_episode':
        if step_scores_intermediate is not None:
            if all(score == 0 for score in step_scores_intermediate):
                return no_update
            else:
                feedback = step_scores_intermediate
                e_idx = last_labeled_episode['episode_idx']
                p_idx = last_labeled_episode['proc_idx']
                b_idx = last_labeled_episode['buffer_idx']
                mean_reward = last_labeled_episode['mean_reward']
                mean_loss = last_labeled_episode['mean_loss']
                entropy = last_labeled_episode['entropy']

                # add data to user_data csv:
                data = [time,
                        str(p_idx),
                        str(e_idx),
                        str(b_idx),
                        'cn',
                        json.dumps(feedback),
                        mean_reward,
                        mean_loss,
                        entropy]
                write_user_data_feedback(data, session_data_filename)

                print(f'feedback: {feedback}, {p_idx}, {e_idx}, {b_idx}')

                return feedback, p_idx, e_idx, b_idx
        else:
            return no_update

    # if the c_submit_button_new has been clicked:
    if trigger_id == 'c_submit_button_new':
        feedback = step_scores

        # add data to user_data csv:
        data = [time, str(proc_ids[episode]), str(episode_ids[episode]), 'cn', json.dumps(feedback)]
        write_user_data_feedback(data, session_data_filename)

        print(f'{feedback}, {proc_ids[episode]}, {episode_ids[episode]}')

        return feedback, proc_ids[episode], episode_ids[episode]

    # if the critique submit button has been clicked:
    if trigger_id == 'c_submit_button':
        print('critique submit button clicked#################################################')
        # if no feedback has been given:
        if colors == ['#c6c3c2']:
            return no_update
        # else derive feedback from colors:
        else:
            # if the first slider value is not == 0 but < 1 we have to add one segment with length 1 to the segments
            # because of calculation of diffs (see line 176f):
            if len(c_slider_values) > 0 and 0 < c_slider_values[0] < 1:
                single_step = True
            else:
                single_step = False
            # round slider values to next smaller integers:
            c_slider_values = [int(c_slider_value) for c_slider_value in c_slider_values]

            # add min and max to slider values if not already there:
            if 0 not in c_slider_values:
                c_slider_values.insert(0, 0)
            if c_slider_max not in slider_values:
                c_slider_values.append(c_slider_max)

            # get length of each feedback segment:
            segment_lengths = diffs(c_slider_values)
            if single_step:
                segment_lengths.insert(0, 1)
            else:
                # have to add 1 to the first segment length because the first segment starts at 0:
                segment_lengths[0] += 1

            feedback = []
            for i, color in enumerate(colors):
                if color == '#c6c3c2':  # grey
                    score = 0
                elif color == '#B3E5B5':  # green
                    score = 1
                else:
                    score = -1  # red

                # add score to feedback list as many times as the segment length:
                feedback.extend([score] * segment_lengths[i])

            # add data to user_data csv:
            data = [time, str(proc_ids[episode]), str(episode_ids[episode]), 'c', json.dumps(feedback)]
            write_user_data_feedback(data, session_data_filename)

            print(f'{feedback}, {proc_ids[episode]}, {episode_ids[episode]}')
            return feedback, proc_ids[episode], episode_ids[episode]

    # if the natural language submit button has been clicked:
    # if there are more segments (i.e. sentences) than steps, cut off excess segments:
    if len(feedback_segments) > len(actions[episode]):
        feedback_segments = feedback_segments[:len(actions[episode])]

    # get sentiment scores of the feedback segments:
    sentiment_scores = [np.round(np.mean(sentiment_analysis(fs)[1]), 2) for fs in feedback_segments]

    # if the sentiment analysis has no scores, abort giving feedback:
    if np.isnan(sentiment_scores).any():
        return no_update

    # apply simple segmentation, i.e. evenly map sentiment scores on steps:
    if segmentation_mode == 'simple':
        # get number of steps per segment:
        steps_per_segment = len(actions[episode]) // len(feedback_segments)

        # get number of remaining steps:
        remaining_steps = len(actions[episode]) % len(feedback_segments)

        # generate list of feedback scores:
        feedback = [s for s in sentiment_scores for _ in range(steps_per_segment)]
        # append scores for remaining steps:
        feedback = feedback + [sentiment_scores[-1]] * remaining_steps

        return feedback, proc_ids[episode], episode_ids[episode]

    # apply manual segmentation, i.e. map sentiment scores on steps according to slider values:
    if segmentation_mode == 'manual':
        # cut off excess slider values:
        slider_values = slider_values[:len(sentiment_scores) - 1]
        if slider_max in slider_values:
            slider_values.remove(slider_max)
        slider_values = list(map(math.ceil, slider_values))
        if slider_min not in slider_values:
            slider_values.insert(0, slider_min)
        slider_values.append(slider_max + 1)

        # generate list of feedback scores:
        feedback = []
        # append scores to list according to manual segmentation:
        for i, v in enumerate(sentiment_scores):
            if i == len(sentiment_scores) - 1:
                scores_temp = [v] * (slider_max + 1 - slider_values[i])
            else:
                scores_temp = [v] * (slider_values[i + 1] - slider_values[i])
            feedback = feedback + scores_temp

        # add data to user_data csv:
        data = [time, str(proc_ids[episode]), str(episode_ids[episode]), 'nl', json.dumps(feedback)]
        write_user_data_feedback(data, session_data_filename)

        return feedback, proc_ids[episode], episode_ids[episode]
    return no_update
