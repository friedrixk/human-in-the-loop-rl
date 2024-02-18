import dash
from dash import html, dcc, Input, Output, callback, State, no_update
from plotly import graph_objects as go
from .loss_reward import default_fig0, default_fig1
import dash_bootstrap_components as dbc
from PIL import Image
from app.critique_new import critique_layout_new
from app.data_writer import write_user_data_interactions
import datetime
from gym_minigrid.minigrid import Grid
import numpy as np

actions_list = ['turn left', 'turn right', 'move forward', 'pickup object', 'drop object', 'toggle door/switch', 'done']

# dict to decode actions from int to text
actions_dict = {0: 'turn left', 1: 'turn right', 2: 'move forward', 3: 'pickup object', 4: 'drop object',
                5: 'toggle door/switch', 6: 'done'}

# dict to map agent direction to the kind of marker used in the grid graph
marker_dict = {0: 'triangle-right', 1: 'triangle-down', 2: 'triangle-left', 3: 'triangle-up'}

arrows_fullscreen_icon = html.I(className='bi bi-arrows-fullscreen', style={'font-size': '2vh'})
refresh_plot_icon = html.I(className='bi bi-arrow-clockwise', style={'font-size': '1.5vh'})

color_scale = {
    'turn left': 'red',
    'turn right': 'red',
    'move forward': 'red',
    'pickup object': '#1c87ff',
    'drop object': '#1c87ff',  # '#ffed5d',
    'toggle door/switch': 'black',
    'done': 'rgb(0, 210,0)'
}

# dict to map agent action to the kind of marker used in the grid graph:
style_scale = {
    'turn left': 'triangle-left',
    'turn right': 'triangle-right',
    'move forward': 'triangle-up',
    'pickup object': 'star-triangle-up',  # 'diamond-tall',
    'drop object': 'star-triangle-down',  # 'diamond-wide',
    'toggle door/switch': 'line-nw',
    'done': 'circle'
}

# list of images of markers of the actions:
action_imgs_paths = ['app/imgs/turn_left.png', 'app/imgs/turn_right.png', 'app/imgs/move_forward.png',
                     'app/imgs/pickup_object.png', 'app/imgs/drop_object.png', 'app/imgs/toggle_door_switch.png',
                     'app/imgs/done.png']

action_imgs = [Image.open(path) for path in action_imgs_paths]

shifts = {
    0: (3, 0),
    1: (0, 3),
    2: (-3, 0),
    3: (0, -3)
}


def actions_to_text(ints):
    """
    function to decode actions from int to text
    @type ints: list of integers
    """
    return [actions_dict[i] for i in ints]


# layout:
uncertainty = html.Div([
    dcc.Store('episode_imgs'),
    dcc.Store('entropy_stepwise'),
    dcc.Store('entropy_episode_wise'),
    dcc.Store('selected_steps'),
    dcc.Store(id='agent_x_pos'),
    dcc.Store(id='agent_y_pos'),
    dcc.Store(id='agent_dir'),
    dcc.Store(id='grid_width'),
    dcc.Store(id='grid_height'),
    dcc.Store(id='display_step'),
    dcc.Store(id='imgs'),
    dcc.Store(id='actions'),
    dcc.Store(id='missions'),
    dcc.Tooltip('grid_tooltip'),
    dcc.Store(id='uncertainty_bg_colors', data=['#c6c3c2']),  # grey
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    # label that holds the mission of the currently selected episode:
                    html.Label(id='mission',
                               style={'padding-left': '0.5vw', 'font-weight': 'bold'})
                ], style={'margin-top': '1vh', 'margin-left': '0.5vw', 'height': '3vh', 'width': '69.5vw',
                          'background-color': '#EFF6FB'}),
            ], style={'display': 'flex', 'height': '3vh', 'width': '69.5vw', 'background-color': 'white'}),
            html.Div([
                critique_layout_new,
                dbc.Button([refresh_plot_icon, 'reset plot'], id='reset_uncertainty_button', n_clicks=1,
                           style={'margin-left': '0.5vw', 'height': '3vh', 'margin-top': '1.75vh', 'line-height': '3vh',
                                  'margin-right': '0.5vw',
                                  'font-family': 'Open Sans, verdana, arial, sans-serif !important',
                                  'textTransform': 'none', 'padding': '0vh 0.2vw 0vh 0.2vw'}),
            ], style={'display': 'flex', 'height': '3vh', 'width': '67vw', 'background-color': 'white'}),
            # uncertainty graph:
            dcc.Graph(id='uncertainty_graph', figure=default_fig0, clear_on_unhover=False,
                      style={'padding-top': '2vh', 'height': '32.5vh', 'width': '70.5vw', 'margin-right': '0.5vw'},
                      config={'displaylogo': False, 'displayModeBar': True, 'scrollZoom': True,
                              'modeBarButtonsToRemove': ['select2d', 'zoom2d', 'zoomIn', 'zoomOut',
                                                         'toImage', 'resetScale2d']}),
        ], style={'display': 'inline', 'background-color': 'white'}),
        html.Div([
            # button to reset the grid graph, content is aligned to the right:
            # html.Div([
            #     dbc.Button([refresh_plot_icon, ' reset grid'], id='reset_grid_button', n_clicks=1,
            #                style={'height': '3vh', 'margin-top': '1vh', 'line-height': '3vh',
            #                       'font-family': 'Open Sans, verdana, arial, sans-serif !important',
            #                       'textTransform': 'none', 'padding': '0vh 0.2vw 0vh 0.2vw', 'margin-right': '0.5vw'}),
            # ], style={'display': 'flex', 'justify-content': 'flex-end', 'height': '3vh', 'width': '20.5vw'}),
            # grid graph:
            dcc.Graph(id='grid_graph', figure=default_fig0,
                      style={'height': '40.5vh', 'width': '24vw', 'padding-top': '0vh', 'margin-top': '0vh'},
                      config={'displayModeBar': False, 'scrollZoom': True}),
        ], style={'width': '20.5vw', 'height': '41vh', 'background-color': 'white'})
    ], style={'display': 'flex', 'height': '41vh', 'width': '95vw'})
])


@callback(Output('uncertainty_graph', 'figure'),
          Input('uncertainty_display_episode', 'data'),
          Input('start_button', 'n_clicks'),
          Input('stop_button', 'n_clicks'),
          Input('display_step', 'data'),
          Input('slider_values', 'data'),
          Input('uncertainty_bg_colors', 'data'),
          Input('critique_graph_hoverData', 'data'),
          Input('reset_uncertainty_button', 'n_clicks'),
          Input('step_scores', 'data'),
          State('slider_max', 'data'),
          State('entropy_stepwise', 'data'),
          State('actions', 'data'),
          State('agent_dir', 'data'),
          State('feedback_tabs', 'value'),
          config_prevent_initial_callbacks=True
          )
def update_uncertainty_graph(episode, start_click, stop_click, displaystep, slider_values, colors, hoverdata,
                             reset_click, step_scores, maximum, entropy_stepwise, actions, directions, feedback_tab):
    # get id of component that triggered the callback:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # when training has only started and there is no data available yet:
    # if trigger_id == 'start_button':
    #     return default_fig1

    fig = go.Figure()
    if episode is not None:

        # if a certain step has been selected, highlight it:
        # if displaystep is not None:
        #     # this trace only displays the single data point of the currently selected step:
        #     selected_step_trace = go.Scatter(x=[displaystep],
        #                                      y=[entropy_stepwise[episode][displaystep]],
        #                                      marker=dict(size=20, color='rgba(91, 91, 91, 0.5)', ),
        #                                      #  line=dict(width=20, color='rgba(91, 91, 91, 0.5)')),
        #                                      mode='markers',
        #                                      hoverinfo='skip',
        #                                      showlegend=False)

        # adding annotation in the top right corner of the graph to show data of the currently selected step
        # fig.add_annotation(x=1,
        #                    y=1,
        #                    xref='paper',
        #                    yref='paper',
        #                    xanchor='right',
        #                    yanchor='top',
        #                    text=
        #                    f'<b>selected step: {displaystep}' +
        #                    f'<br><b>uncertainty: {round(entropy_stepwise[episode][displaystep], 2)}',
        #                    #  + f'<br>action: {actions_to_text(actions[episode])[displaystep]}',
        #                    font=dict(
        #                        family='Open Sans, verdana, arial, sans-serif',
        #                        size=12,
        #                        color="black"
        #                    ),
        #                    showarrow=False,
        #                    align='left',
        #                    bgcolor='rgba(239, 246, 251, 1)',
        #                    )
        # fig.add_traces([selected_step_trace])

        # setting the colors for each marker:
        action_colors = [color_scale[action] for action in actions_to_text(actions[episode])]

        # setting the marker styles for each marker:
        # styles = [marker_dict[dir] for dir in directions[episode]]  # when using direction as marker style
        styles = [style_scale[action] for action in
                  actions_to_text(actions[episode])]  # when using action as marker style

        entropy_trace = go.Scatter(name='uncertainty',
                                   x=list(range(0, len(entropy_stepwise[episode]))),
                                   y=entropy_stepwise[episode],
                                   line=dict(color='red', width=1),
                                   # marker=dict(color=action_colors, size=15,
                                   #             symbol=styles, line=dict(width=1, color='black')),
                                   mode='lines',
                                   # customdata=actions_to_text(actions[episode]),
                                   # hovertemplate=
                                   # '%{y:.2f}',  # + '<br>action: %{customdata}',
                                   hoverinfo='skip',
                                   yaxis='y1',
                                   showlegend=False,
                                   )
        fig.add_traces([entropy_trace])

        # Create a trace for each action (here, all markers belong to the same trace):
        # for action in actions_list:
        #     action_colors = [color_scale[action]]
        #     styles = [style_scale[action]]
        #     trace = go.Scatter(x=[0], y=[-1], mode='markers', name=action,
        #                        marker=dict(color=action_colors, symbol=styles, size=15,
        #                                    line=dict(width=1, color='black')))
        #     fig.add_traces([trace])

        # create a trace for each action:
        for a in actions_list:
            # create a trace for each action type:
            # the indices of the steps of the action type:
            x = [i for i, action in enumerate(actions_to_text(actions[episode])) if action == a]
            # the uncertainty values of the move forward actions:
            y = [entropy_stepwise[episode][i] for i in x]
            # setting marker:
            marker = dict(color=color_scale[a], size=12, symbol=style_scale[a],
                          line=dict(width=1, color='black'))
            # creating the trace:
            trace = go.Scatter(name=a,
                               x=x,
                               y=y,
                               marker=marker,
                               mode='markers',
                               hoverinfo='skip',
                               yaxis='y1',
                               # customdata=actions_to_text(actions[episode]),
                               hovertemplate=
                               '%{y:.2f}',  # + '<br>action: %{customdata}',
                               )
            # add the trace to the figure:
            fig.add_traces([trace])

        # adding a vertical line for each handle of the range slider in the feedback component to visualize the segments
        if slider_values:
            for value in slider_values:
                fig.add_vline(x=value, line_width=1, line_dash="dash", line_color="gray")
        else:
            slider_values = [0, len(entropy_stepwise[episode]) - 1]

        # if feedback-tab is not == 'tab-3':
        if feedback_tab != 'tab-3':
            # adding a vertical rectangle for each step sequence with the corresponding color:
            if 0 not in slider_values:
                slider_values.insert(0, 0)
            if len(entropy_stepwise[episode]) - 1 not in slider_values:
                slider_values.append(len(entropy_stepwise[episode]) - 1)
            for i in range(len(slider_values) - 1):
                x0 = slider_values[i]
                x1 = slider_values[i + 1]
                fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[i], opacity=0.5, layer='below', line_width=0)

        # add a horizontal rectangle for the currently hovered step sequence in critique graph:
        if hoverdata:
            # get curve number:
            curvenumber = hoverdata['points'][0]['curveNumber']
            x0 = slider_values[curvenumber]
            x1 = slider_values[curvenumber + 1]
            fig.add_vrect(x0=x0, x1=x1, layer='above', line_width=2, line_color='rgb(133, 133, 133)')

        # if feedback tab is 'tab-3' add a vertical rectangle for every step according to the step_scores:
        if feedback_tab == 'tab-3':

            # generate a list of rectangles by combining neighboring steps with same score (either 1 or -1) to one
            # rectangle where x0 is the first step of the rectangle -0.5 and x1 is the last step of the rectangle +0.5:
            rectangles = []
            i = 0

            while i < len(step_scores):
                if step_scores[i] != 0:
                    start_index = i
                    current_score = step_scores[i]

                    while i < len(step_scores) - 1 and step_scores[i + 1] == current_score:
                        i += 1

                    end_index = i
                    rectangles.append([start_index - 0.5, end_index + 0.5, current_score])

                i += 1

            # add the rectangles to the figure:
            for rectangle in rectangles:
                if rectangle[2] == 1:
                    fig.add_vrect(x0=rectangle[0], x1=rectangle[1], opacity=0.2, layer='below', line_width=0,
                                  fillcolor="green")
                elif rectangle[2] == -1:
                    fig.add_vrect(x0=rectangle[0], x1=rectangle[1], opacity=0.2, layer='below', line_width=0,
                                  fillcolor="red")

            # step_scores = np.array(step_scores)
            #
            # for i, score in enumerate(step_scores):
            #     if score == 1:
            #         fig.add_vrect(x0=i - 0.5, x1=i + 0.5, opacity=0.2, layer='below', line_width=0, fillcolor="green")
            #     elif score == -1:
            #         fig.add_vrect(x0=i - 0.5, x1=i + 0.5, opacity=0.2, layer='below', line_width=0, fillcolor="red")

        fig.update_layout(title='agent uncertainty',
                          xaxis=dict(title={'text': 'steps', 'standoff': 0}, gridcolor='rgb(229, 236, 246)',
                                     showgrid=True, range=[0 - 0.01 * len(entropy_stepwise[episode]),
                                                           len(entropy_stepwise[episode]) - 1 + 0.01 * len(
                                                               entropy_stepwise[episode])], autorange=False),
                          yaxis=dict(title='uncertainty', range=[-0.05, 1.05], autorange=False, fixedrange=True,
                                     gridcolor='rgb(229, 236, 246)'),
                          # yaxis2=dict(title='KL divergence',
                          #             side='right', gridcolor='rgb(229, 236, 246)'),
                          margin=dict(l=5, r=5, t=30, b=0),
                          plot_bgcolor='white',
                          hovermode='x unified',
                          dragmode='pan',
                          clickmode='event+select'
                          )

        # this is needed to avoid the UI state to be reset when a marker has been clicked (e.g. when zoomed in and
        # clicking on a step the axes will be reset which we don't want in this case.
        fig['layout']['uirevision'] = reset_click  # tsne_selection

    # if no episode is selected give the user a hint:
    else:
        fig.update_layout(title='',
                          xaxis=dict(visible=False, range=[0, 600], autorange=False),
                          yaxis=dict(visible=False, range=[0, 2], autorange=False, ),
                          margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor='#f9f9f9',
                          annotations=[{'text': 'Select an episode to display data.',
                                        'xref': 'paper',
                                        'yref': 'paper',
                                        'showarrow': False,
                                        'font': {'size': 12}
                                        }]
                          )

    return fig


# @callback(
#     Output('uncertainty_graph', 'clickData'),
#     Output('display_step', 'data'),
#     Input('uncertainty_graph', 'clickData'),
#     Input('uncertainty_display_episode', 'data'),
#     State('display_step', 'data'),
#     config_prevent_initial_callbacks=True
# )
# def update_display_step(clickdata, episode, step):
#     """
#     Update the currently selected step.
#     @param clickdata: data returned when clicking on a data point in uncertainty_graph
#     @param episode: the index of the currently selected episode
#     @param step: the index of the currently selected step
#     @return: index of the newly selected step or None
#     """
#     # get id of component that triggered the callback:
#     ctx = dash.callback_context
#     trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
#
#     # when the selected episode changes, display_step shall be reset to None:
#     if trigger_id == 'uncertainty_display_episode':
#         return None, None
#
#     if clickdata:
#         x = clickdata['points'][0]['x']
#         if x == step:  # unselect the currently selected step
#             step = None
#         else:
#             step = x  # set new currently selected step
#     return None, step


@callback(
    Output('grid_graph', 'figure'),
    Input('display_step', 'data'),
    Input('uncertainty_graph', 'hoverData'),
    Input('stop_button', 'n_clicks'),
    Input('start_button', 'n_clicks'),
    # Input('reset_grid_button', 'n_clicks'),
    State('uncertainty_display_episode', 'data'),
    Input('uncertainty_graph', 'clickData'),
    State('agent_x_pos', 'data'),
    State('agent_y_pos', 'data'),
    State('agent_dir', 'data'),
    State('grid_width', 'data'),
    State('grid_height', 'data'),
    Input('episode_imgs', 'data'),
    State('actions', 'data'),
    config_prevent_initial_callbacks=True
)
def update_grid_graph(displaystep, hoverdata, stop_click, start_click, episode, clickData, x, y, direction,
                      grid_width, grid_height, episode_imgs, actions):
    # get id of component that triggered the callback:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # when training has only started and there is no data available yet:
    # if trigger_id == 'start_button':
    #     return default_fig1

    fig = go.Figure()
    if episode is not None:

        # (temporarily) change display_step according to hoverdata if callback has been triggered by hovering
        # uncertainty_graph:
        if hoverdata is not None:
            hovered_points = [p['x'] for p in hoverdata['points']]
            displaystep = max(hovered_points)
        # elif clickData is not None:
        #     displaystep = clickData['points'][0]['x']

        if displaystep is not None and displaystep < len(x[episode]):
            x_part = x[episode][:displaystep + 1]
            y_part = y[episode][:displaystep + 1]

            mode = 'lines'

            # workaround to make the line of the partial trace disappear if there is only one (x,y) in the trace
            if all(x == x_part[0] for x in x_part) and all(y == y_part[0] for y in y_part):
                mode = 'markers'

            # the trace that shows the trajectory of the agent up to the currently selected step
            partial_trace = go.Scatter(name='partial trajectory',
                                       x=x_part,
                                       y=y_part,
                                       line=dict(color='red'),
                                       mode=mode,
                                       hoverinfo='skip',
                                       showlegend=False,
                                       yaxis='y1')

            # preparing marker direction:
            # marker = marker_dict[direction[episode][displaystep]]

            # the trace that shows the current position and direction of the agent (it only holds one single data point)
            # selected_step_trace = go.Scatter(x=[x[episode][displaystep]],
            #                                  y=[y[episode][displaystep]],
            #                                  marker=dict(symbol=marker, size=15, color='red',
            #                                              line=dict(color='black', width=1)),
            #                                  mode='markers',
            #                                  hoverinfo='skip',
            #                                  showlegend=False)

            # prepare shift of the observation rectangle that is displayed in front of the agent:
            x_shift, y_shift = shifts[direction[episode][displaystep]]

            # adding a 7x7 rectangle in front of the agent to represent the agents current observation;
            # depending on the agent's direction, the rectangle is shifted in the corresponding direction:
            # fig.add_shape(type="rect",
            #               x0=x[episode][displaystep] - 3.5 + x_shift,
            #               y0=y[episode][displaystep] - 3.5 + y_shift,
            #               x1=x[episode][displaystep] + 3.5 + x_shift,
            #               y1=y[episode][displaystep] + 3.5 + y_shift,
            #               line=dict(color="RoyalBlue", width=1),
            #               fillcolor="LightSkyBlue",
            #               opacity=0.5,
            #               layer="above",
            #               yref='y1')

            observation_rect = go.Scatter(x=[x[episode][displaystep] - 3.5 + x_shift,
                                             x[episode][displaystep] + 3.5 + x_shift,
                                             x[episode][displaystep] + 3.5 + x_shift,
                                             x[episode][displaystep] - 3.5 + x_shift,
                                             x[episode][displaystep] - 3.5 + x_shift],
                                          y=[y[episode][displaystep] - 3.5 + y_shift,
                                             y[episode][displaystep] - 3.5 + y_shift,
                                             y[episode][displaystep] + 3.5 + y_shift,
                                             y[episode][displaystep] + 3.5 + y_shift,
                                             y[episode][displaystep] - 3.5 + y_shift],
                                          mode='lines',
                                          line=dict(color='#e0e0e0', width=1),
                                          fill='toself',
                                          fillcolor='#e0e0e0',
                                          opacity=0.37,
                                          yaxis='y1',
                                          hoverinfo='skip',
                                          showlegend=False)
            fig.add_trace(observation_rect)

            fig.add_traces([partial_trace])  # selected_step_trace

            # adding image of the current action:
            fig.add_layout_image(dict(source=action_imgs[actions[episode][displaystep]],
                                      xref='paper',
                                      yref='paper',
                                      x=0.9,#x[episode][displaystep] + 0.5,
                                      y=0.9,#y[episode][displaystep] - 0.5,
                                      sizex=0.1,
                                      sizey=0.1,
                                      sizing='stretch',
                                      layer='above',
                                      xanchor='left',
                                      yanchor='bottom'))

            # adding annotation with the corresponding action of displaystep:
            # fig.add_annotation(x=x[episode][displaystep],
            #                    y=y[episode][displaystep],
            #                    text=f'<img src="{image_path}" width="50" height="50"><br>{actions_to_text([actions[episode][displaystep]])[0]}',
            #                    # showarrow=True,
            #                    # arrowhead=1,
            #                    # ax=0,
            #                    # ay=-40,
            #                    font=dict(
            #                        family='Open Sans, verdana, arial, sans-serif',
            #                        size=12,
            #                        color="black"
            #                    ),
            #                    bgcolor='white')

        # opaque trace for the whole trajectory
        whole_trace = go.Scatter(name='whole trajectory',
                                 x=x[episode],
                                 y=y[episode],
                                 line=dict(color='red'),
                                 mode='lines',
                                 opacity=0.5,
                                 hoverinfo='skip',
                                 showlegend=False,
                                 yaxis='y1')

        # only add whole_trajectory if there is more than one distinct (x,y) in the trace:
        if len(set(x[episode])) > 1 or len(set(y[episode])) > 1:
            fig.add_trace(whole_trace)

        fig.update_layout(title='',
                          xaxis=dict(title='',
                                     gridcolor='rgb(229, 236, 246)',
                                     showgrid=False,
                                     zeroline=False,
                                     range=[-0.5, grid_width[0][0] - 0.5],
                                     tickmode='linear',
                                     tick0=-0.5,
                                     dtick=1,
                                     showticklabels=False),
                          yaxis=dict(title='',
                                     showgrid=False,
                                     zeroline=False,
                                     gridcolor='rgb(229, 236, 246)',
                                     range=[grid_height[0][0] - 0.5, -0.5],
                                     tickmode='linear',
                                     tick0=-0.5,
                                     dtick=1,
                                     showticklabels=False),
                          plot_bgcolor='white',
                          margin=dict(l=5, r=5, t=30, b=0),
                          dragmode='pan',
                          )

        # Adding image of the episodes first frame as background image
        if displaystep is not None and displaystep < len(x[episode]):
            grid, _ = Grid.decode(np.array(episode_imgs[displaystep]))
            img = grid.render(
                32,
                (x[episode][displaystep], y[episode][displaystep]),
                direction[episode][displaystep],
                highlight_mask=None  # highlight_mask if highlight else None
            )

        else:
            grid, _ = Grid.decode(np.array(episode_imgs[0]))
            img = grid.render(
                32,
                (x[episode][0], y[episode][0]),
                direction[episode][0],
                highlight_mask=None  # highlight_mask if highlight else None
            )

        image = Image.fromarray(img)
        fig.add_layout_image(dict(source=image,
                                  xref='x',
                                  yref='y',
                                  x=-0.5,
                                  y=-0.5,
                                  sizex=grid_width[0][0],
                                  sizey=grid_height[0][0],
                                  sizing='stretch',
                                  layer='below'))

        fig['layout']['uirevision'] = 'anything'  # reset_click

    # if no episode is selected give the user a hint:
    else:
        fig.update_layout(title='',
                          xaxis=dict(visible=False),
                          yaxis=dict(visible=False),
                          margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor='#f9f9f9',
                          annotations=[{'text': 'Select an episode to display data.',
                                        'xref': 'paper',
                                        'yref': 'paper',
                                        'showarrow': False,
                                        'font': {'size': 12}
                                        }]
                          )

    # if clickData is not None:
    #     print('clickdata is not none')
    #     fig.data[0].selectedpoints = None

    return fig


@callback(
    Output('mission', 'children'),
    Input('uncertainty_display_episode', 'data'),
    Input('missions', 'data'),
    Input('start_button', 'n_clicks')
)
def update_display_mission(episode, missions, start_click):
    # get id of component that triggered the callback:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'start_button':
        return 'Mission: No episode selected.'

    if episode is not None:
        mission = f'Mission: {missions[episode]}'
    else:
        mission = 'Mission: No episode selected.'
    return mission


@callback(Output('uncertainty_bg_colors', 'data'),
          Input('critique_segment_colors', 'data'),
          Input('nl_segment_colors', 'data'),
          Input('feedback_tabs', 'value'),
          Input('slider_values', 'data'),
          config_prevent_initial_callbacks=True)
def update_uncertainty_bg_colors(critique_segment_colors, nl_segment_colors, feedback_type, slider_values):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'feedback_tabs':
        # if slider_values is None or slider_values == []:
        #     print('triggered by slider values is None')
        #     return ['#c6c3c2']  # grey
        # else:
        #     print('wrong turn taken')
        return ['#c6c3c2']  # grey
    elif feedback_type == 'tab-1' and critique_segment_colors != [] and critique_segment_colors is not None:
        return critique_segment_colors
    elif feedback_type == 'tab-2':
        if nl_segment_colors is not None and nl_segment_colors != []:
            return nl_segment_colors
        else:
            return ['#c6c3c2'] * (len(slider_values) + 1)  # grey
    else:
        return ['#c6c3c2']  # grey


# callback that prints the index of the points that have been selected with the lasso tool to the screen:
@callback(Output('selected_steps', 'data'),
          Input('uncertainty_graph', 'selectedData'),
          Input('uncertainty_display_episode', 'data'),
          )
def print_lasso_selection(selectedData, episode):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'uncertainty_display_episode':
        return None

    elif selectedData is not None:
        selectedpoints = sorted(selectedData['points'], key=lambda k: k['x'])
    else:
        selectedpoints = None
    if not selectedpoints:
        return None
    return selectedpoints


# callback to update n_clicks of reset plot button (needed to reset uncertainty graph when episode is changed):
@callback(Output('reset_uncertainty_button', 'n_clicks'),
          Input('uncertainty_display_episode', 'data'),
          State('reset_uncertainty_button', 'n_clicks'),
          config_prevent_initial_callbacks=True)
def update_reset_uncertainty_button(episode, n_clicks):
    return n_clicks + 1


# callback to set clickData of uncertainty graph to None (needed when data is refreshed):
# @callback(Output('uncertainty_graph', 'clickData'),
#           Input('refresh_button', 'n_clicks'),
#           Input('uncertainty_display_episode', 'data'),
#           config_prevent_initial_callbacks=True)
# def update_uncertainty_graph_clickData(n_clicks, episode):
#     return None


# callback to print hoverdata and selecteddata and clickdata whenever uncertainty graph is used:
@callback(Output('uncertainty_graph', 'hoverData'),
          Input('uncertainty_graph', 'hoverData'),
          Input('uncertainty_graph', 'selectedData'),
          Input('uncertainty_graph', 'clickData'),
          State('session_interaction_data_filename', 'data'),
          State('uncertainty_display_episode', 'data'),
          State('episode_ids', 'data'),
          State('proc_ids', 'data'),
          State('buffer_ids', 'data'),
          config_prevent_initial_callbacks=True)
def write_select_and_hover_data(hoverData, selectedData, clickData, filename, episode, episode_ids, proc_ids, b_ids):
    ctx = dash.callback_context

    if filename is not None:
        # timestamp for logging
        time = datetime.datetime.now().strftime("%H:%M:%S")
        if episode is not None:
            e_idx = episode_ids[episode]
            p_idx = proc_ids[episode]
            b_idx = b_ids[episode]
        else:
            e_idx = None
            p_idx = None
            b_idx = None

        data = None

        if ctx.triggered[0]["prop_id"] == 'uncertainty_graph.hoverData':
            if hoverData is not None:
                step_idx = hoverData['points'][0]['x']
                data = [time, 'uncertainty', 'hover', str(e_idx), str(p_idx), str(step_idx), str(b_idx), str(hoverData)]
        elif ctx.triggered[0]["prop_id"] == 'uncertainty_graph.selectedData':
            if selectedData is not None:
                point_ids = [point['x'] for point in selectedData['points']]
                data = [time, 'uncertainty', 'select', str(e_idx), str(p_idx), 'see details', str(b_idx),
                        str(point_ids)]
        else:
            if clickData is not None:
                step_idx = clickData['points'][0]['x']
                data = [time, 'uncertainty', 'click', str(e_idx), str(p_idx), str(step_idx), str(b_idx), str(clickData)]
        if data is not None:
            write_user_data_interactions(data, filename)
    return no_update


# callback for updating 'episode_images' when uncertainty_display_episode is changed:
@callback(Output('episode_imgs', 'data'),
          Input('uncertainty_display_episode', 'data'),
          Input('imgs', 'data'))
def update_episode_imgs(episode, imgs):
    if episode is None:
        return None
    return imgs[episode]
