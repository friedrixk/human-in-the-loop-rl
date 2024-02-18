import dash
from dash import html, dcc, Input, Output, callback, State, no_update
import plotly.graph_objs as go
import plotly.express as px
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from app.data_writer import write_user_data_interactions
import datetime

env_name = 'env_placeholder'

# default figure for when training has not been started yet:
default_fig0 = go.Figure()
default_fig0.update_layout(title='',
                           xaxis=dict(visible=False),
                           yaxis=dict(visible=False),
                           margin=dict(l=10, r=10, t=10, b=10),
                           plot_bgcolor='#f9f9f9',
                           annotations=[{'text': 'Start training to display data.',
                                         'xref': 'paper',
                                         'yref': 'paper',
                                         'showarrow': False,
                                         'font': {'size': 12}
                                         }]
                           )

# default figure for when training has started but no data is available yet.
default_fig1 = go.Figure()
default_fig1.update_layout(title='',
                           xaxis=dict(visible=False),
                           yaxis=dict(visible=False),
                           margin=dict(l=10, r=10, t=10, b=10),
                           plot_bgcolor='#f9f9f9',
                           annotations=[{'text': 'Wait until data is available.<br>Then click "refresh" button.',
                                         'xref': 'paper',
                                         'yref': 'paper',
                                         'showarrow': False,
                                         'font': {'size': 12}
                                         }]
                           )

# layout
loss_reward = html.Div([
    dcc.Store(id='uncertainty_display_episode'),
    dcc.Store(id='selected_points'),
    dcc.Store(id='loss_episode_wise'),
    dcc.Store(id='rewards_episode_wise'),
    dcc.Store(id='episodes_fail'),
    dcc.Store(id='episodes_success'),
    dcc.Store(id='feedback_episodes'),
    html.Div([
        html.Div([
            html.Div([
                html.Div([], style={'width': '26vw'}),
                html.Div([], style={'width': '26px'}),
                html.Label('Select episodes: '),
                # radio items for episode selection:
                dcc.RadioItems(id='episode_opt', options=['all', 'success episodes', 'fail episodes'], value='all',
                               inline=True, style={'text-align': 'center', 'background-color': 'white'})
            ], style={'display': 'flex', 'width': '95vw', 'background-color': 'white', 'height': '3vh'}),
        ], style={'display': 'flex', 'height': '3vh'}),
        html.Div([
            html.Div([
                # dcc.Loading([
                # tsne graph:
                dcc.Graph(id='tSNE_graph', figure=default_fig0, style={'height': '39vh', 'width': '26vw'},
                          clear_on_unhover=True, config={'displayModeBar': True, 'scrollZoom': True,
                                                         'modeBarButtonsToRemove': ['select2d', 'zoom2d',
                                                                                    'zoomIn', 'zoomOut',
                                                                                    'select', 'toImage',
                                                                                    'resetScale2d'],
                                                         'displaylogo': False}),
                # ])
            ]),
            # loss and reword graph
            dcc.Graph(id='loss_reward_graph', figure=default_fig0, style={'height': '39vh', 'width': '69vw'},
                      clear_on_unhover=True, config={'displayModeBar': True, 'scrollZoom': True,
                                                     'modeBarButtonsToRemove': ['select2d', 'zoom2d', 'lasso2d', 'pan',
                                                                                'zoomIn', 'zoomOut',
                                                                                'select', 'toImage',
                                                                                'resetScale2d'], 'displaylogo': False}),
        ], style={'display': 'flex'}),
        dcc.Tooltip(id="graph-tooltip", direction='left')
    ], style={'display': 'inline'})
])


@callback(
    Output('selected_points', 'data'),
    Input('tSNE_graph', 'selectedData'),
    Input('episode_opt', 'value'),
    Input('episodes_fail', 'data'),
    Input('episodes_success', 'data'),
    Input('loss_episode_wise', 'data'),
    Input('start_button', 'n_clicks'),
    config_prevent_initial_callbacks=True
)
def update_selected_points(tsneselection, episodeopt, episodes_fail, episodes_success, loss_episode_wise, n_clicks):
    """
    Updates the list of episodes that are selected via the tsne lasso selection or the radio buttons.
    @param tsneselection: dict of the selected points
    @param episodeopt: the selected radio button option from episode_opt
    @param episodes_fail: list of all visualized episodes the agent failed
    @param episodes_success: list of all visualized episodes the agent succeeded
    @param loss_episode_wise: list of loss per episode
    @param n_clicks: number of times the start button has been clicked
    @return:
    """
    print('updating selected_points')

    # get id of component that triggered the callback
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'start_button':
        return no_update

    # set selectedpoints
    selectedpoints = []
    if trigger_id == 'tSNE_graph' and tsneselection is not None:
        for i in range(0, len(tsneselection['points'])):
            selectedpoints.append(tsneselection['points'][i]['pointIndex'])
    elif episodeopt == 'success episodes':
        selectedpoints = episodes_success
    elif episodeopt == 'fail episodes':
        selectedpoints = episodes_fail
    else:
        selectedpoints = list(range(len(loss_episode_wise)))

    return selectedpoints


@callback(Output('loss_reward_graph', 'clickData'),
          Output('uncertainty_display_episode', 'data'),
          State('uncertainty_display_episode', 'data'),
          Input('loss_reward_graph', 'clickData'),
          Input('refresh_button', 'n_clicks'),
          config_prevent_initial_callbacks=True
          )
def uncertainty_display_episode(episode, clickdata, n_clicks):
    """
    This function updates the episode that is displayed in the uncertainty graph. It also resets the clickData parameter
    of loss_reward_graph to None to enable executing the callback when clicking on the same marker again (this is a
    workaround since dash does not support this).
    :param episode: the episode that is displayed in the uncertainty graph
    :param clickdata: the data of the clicked marker
    :param n_clicks: number of times the refresh button has been clicked
    :return: updated list and None
    """
    # get id of component that triggered the callback
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # reset uncertainty_display_episode if vis data is refreshed
    if trigger_id == 'refresh_button':
        return None, None

    elif clickdata:
        x = clickdata['points'][0]['x']
        if x == episode:
            episode = None
        else:
            episode = x
    return None, episode


@callback(
    Output('tSNE_graph', 'figure'),
    Output('tSNE_graph', 'selectedData'),  # return None to ensure that loss-graph is correct after refreshing vis data
    Input('agent_x_pos', 'data'),
    Input('agent_y_pos', 'data'),
    Input('agent_dir', 'data'),
    Input('rewards_episode_wise', 'data'),
    Input('loss_episode_wise', 'data'),
    Input('start_button', 'n_clicks'),
    Input('stop_button', 'n_clicks'),
    State('dictionary', 'data'),
    config_prevent_initial_callbacks=True
)
def update_tsne_figure(x, y, d, rewards, loss, start_click, stop_click, data):
    # get id of component that triggered the callback:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # when training has only started and there is no data available yet or only one data point:
    # if (trigger_id == 'start_button') or len(rewards) < 2:  # and data is None
    #     return default_fig1, None

    df = pd.DataFrame()
    # taking starting position, rewards and loss as features for tsne calculation:
    df['x'] = [item[0] for item in x]
    df['y'] = [item[0] for item in y]
    df['dir'] = [item[0] for item in d]
    df['rewards'] = np.array(rewards)
    df['loss'] = np.array(loss)

    # calculating tsne:
    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(df)

    # generating tsne figure:
    fig = px.scatter(tsne_result,
                     x=0,
                     y=1,
                     color=df.loss,
                     color_continuous_scale='brwnyl')

    fig.update_traces(marker=dict(line=dict(width=0.8, color='black'))),

    fig.update_layout(title='episode similarity',
                      xaxis=dict(title={'text': 'component 1', 'standoff': 5},
                                 gridcolor='rgb(229, 236, 246)', showgrid=True),
                      yaxis=dict(title={'text': 'component 2', 'standoff': 5},
                                 gridcolor='rgb(229, 236, 246)', showgrid=True),
                      margin=dict(l=5, r=5, t=30, b=0),
                      plot_bgcolor='white',
                      coloraxis_colorbar=dict(title='impact', thicknessmode='pixels', thickness=5),
                      dragmode='pan'
                      )
    # fig['layout']['uirevision'] = 'refresh_button'

    return fig, None


@callback(
    Output('loss_reward_graph', 'figure'),
    Input('uncertainty_display_episode', 'data'),
    Input('selected_points', 'data'),
    Input('loss_episode_wise', 'data'),
    Input('rewards_episode_wise', 'data'),
    Input('start_button', 'n_clicks'),
    Input('stop_button', 'n_clicks'),
    Input('feedback_episodes', 'data'),
    State('entropy_episode_wise', 'data'),
    config_prevent_initial_callbacks=True
)
def update_loss_reward_figure(episode, selectedpoints, loss_episode_wise, rewards_episode_wise, start_click,
                              stop_click, feedback_episodes, entropy_episode_wise):
    """
    This function updates the figure of the loss_reward_graph
    :return: the updated figure
    :param episode: the index of the episode that is currently selected and visualized in the uncertainty graph
    :param selectedpoints: the list of episode indexes that are selected via tsne selection or radio button selection
    :param loss_episode_wise: the list of losses of the visualized episodes
    :param rewards_episode_wise: the list of rewards of the visualized episodes
    :param start_click: the number of times the start button has been clicked
    :param stop_click: the number of times the stop button has been clicked
    """
    # get id of component that triggered the callback:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # when training has only started and there is no data available yet:
    # if trigger_id == 'start_button':
    #     return default_fig1

    # setting default (i.e. episode == None) color maps for the traces
    color_map_reward = ['#BAD8ED', ] * 100
    color_map_loss = ['#5b5b5b', ] * 100

    fig = go.Figure()

    # if a certain episode has been selected, highlight it:
    if episode is not None:
        color_map_reward[episode] = '#FFB266'
        color_map_loss[episode] = '#FFB266'

        # this trace only displays the single data point of the currently selected episode:
        selected_episode_trace_loss = go.Scatter(x=[episode],
                                                 y=[loss_episode_wise[episode]],
                                                 marker=dict(size=6, color='#FFB266',
                                                             line=dict(width=8, color='rgba(91, 91, 91, 0.5)')),
                                                 mode='markers+text',
                                                 hoverinfo='skip',
                                                 showlegend=False,
                                                 textposition='top right',
                                                 )

        selected_episode_trace_reward = go.Bar(name='reward',
                                               x=[episode],
                                               y=[rewards_episode_wise[episode]],
                                               marker=dict(color='#FFB266',
                                                           line=dict(width=8, color='rgba(91, 91, 91, 0.5)')),
                                               hoverinfo='skip',
                                               showlegend=False,
                                               yaxis='y2',
                                               selectedpoints=selectedpoints)

        # adding annotation in the top right corner of the graph to show data of the currently selected episode
        fig.add_annotation(x=1,
                           y=1,
                           xref='paper',
                           yref='paper',
                           xanchor='right',
                           yanchor='top',
                           text=f'selected episode: {episode}' +
                                f'<br>impact: {loss_episode_wise[episode]}' +
                                f'<br>reward: {rewards_episode_wise[episode]}',
                           font=dict(
                               family='Open Sans, verdana, arial, sans-serif',
                               size=12,
                               color="black"
                           ),
                           showarrow=False,
                           align='left',
                           bgcolor='rgba(239, 246, 251, 0.5)'
                           )
        fig.add_traces([selected_episode_trace_loss, selected_episode_trace_reward])

    # create trace that highlights episodes that have received feedback (i.e. feedback_episodes[episode] is True):
    if feedback_episodes is not None:
        inds = [index for index, value in enumerate(feedback_episodes) if value]
        print(f'inds: {inds}')
        feedback_given_trace = go.Scatter(x=inds,
                                          y=[1.0 for i in inds],
                                          marker=dict(size=6, color='black', symbol='triangle-down',
                                                      line=dict(width=8, color='black')),
                                          mode='markers+text',
                                          hoverinfo='skip',
                                          showlegend=False,
                                          textposition='top right',
                                          yaxis='y2'
                                          )
        fig.add_traces([feedback_given_trace])

    loss_trace = go.Scatter(name='impact',
                            x=list(range(0, len(loss_episode_wise))),
                            y=loss_episode_wise,
                            marker=dict(color=loss_episode_wise, colorscale='brwnyl',
                                        line=dict(width=0.8, color='black')),
                            # marker=dict(color=color_map_loss),
                            line=dict(color='#5b5b5b', width=0.8),
                            mode='lines + markers',
                            yaxis='y1',
                            selectedpoints=selectedpoints
                            )

    reward_trace = go.Bar(name='reward',
                          x=list(range(0, len(rewards_episode_wise))),
                          y=rewards_episode_wise,
                          marker=dict(color=entropy_episode_wise, colorscale='Bluered', cmin=0, cmax=1,
                                      colorbar=dict(title='uncertainty', thicknessmode='pixels', thickness=5, x=1.2,
                                                    y=0.5)),
                          textposition="outside",
                          yaxis='y2',
                          selectedpoints=selectedpoints)

    y1max = max(loss_episode_wise)
    y1min = min(loss_episode_wise)
    y2max = max(rewards_episode_wise)

    fig.update_layout(title='impact and reward',
                      xaxis=dict(title={'text': 'episodes', 'standoff': 0}, gridcolor='rgb(229, 236, 246)',
                                 showgrid=True),
                      yaxis=dict(title={'text': 'impact', 'standoff': 0},
                                 gridcolor='rgb(229, 236, 246)',
                                 overlaying='y2',
                                 fixedrange=True,
                                 # range=list([y1min - 0.1 * (y1max - y1min), y1max + 0.1 * (y1max - y1min)])),
                                 range=list([-0.05 * y1max, y1max + 0.1 * (y1max - y1min)])),
                      yaxis2=dict(title='reward',
                                  gridcolor='rgb(229, 236, 246)',
                                  side='right',
                                  range=[-0.05, 1],
                                  fixedrange=True),
                      # coloraxis=dict(orientation="h", xanchor='right', x=0, y=0, yanchor='bottom'),
                      # range=list([0, y2max + abs(0.1 * y2max)])),
                      margin=dict(l=5, r=5, t=30, b=0),
                      plot_bgcolor='white',
                      hovermode='x unified',
                      dragmode='pan',
                      barmode='overlay'
                      )
    fig.add_traces([loss_trace, reward_trace])

    # this is needed to avoid the user UI state to be reset when a marker has been clicked (e.g. if only loss trace
    # is displayed and a marker is clicked the other traces will be displayed again)
    fig['layout']['uirevision'] = 'anything'  # tsne_selection

    return fig


# callback to print hoverdata whenever tsne plot is hovered:
@callback(Output('tSNE_graph', 'hoverData'),
          Input('tSNE_graph', 'hoverData'),
          Input('tSNE_graph', 'selectedData'),
          Input('loss_reward_graph', 'hoverData'),
          Input('loss_reward_graph', 'selectedData'),
          Input('loss_reward_graph', 'clickData'),
          State('session_interaction_data_filename', 'data'),
          State('buffer_ids', 'data'),
          Input('uncertainty_display_episode', 'data'),
          State('episode_ids', 'data'),
          State('proc_ids', 'data'),
          config_prevent_initial_callbacks=True)
def write_select_and_hover_data(tsne_hoverData, tsne_selectedData, reward_loss_hoverData, reward_loss_selectedData,
                                reward_loss_clickData, filename, b_ids, episode, e_ids, p_ids):
    ctx = dash.callback_context


    if filename is not None:
        # timestamp for logging
        time = datetime.datetime.now().strftime("%H:%M:%S")

        data = None

        if ctx.triggered[0]["prop_id"] == 'tSNE_graph.hoverData':
            if tsne_hoverData is not None:
                e_idx = e_ids[tsne_hoverData['points'][0]['pointIndex']]
                p_idx = p_ids[tsne_hoverData['points'][0]['pointIndex']]
                b_idx = b_ids[tsne_hoverData['points'][0]['pointIndex']]
                data = [time, 'tSNE', 'hover', str(e_idx), str(p_idx), 'None', str(b_idx), str(tsne_hoverData)]
        elif ctx.triggered[0]["prop_id"] == 'tSNE_graph.selectedData':
            if tsne_selectedData is not None:
                point_ids = [point['pointIndex'] for point in tsne_selectedData['points']]
                data = [time, 'tSNE', 'select', 'undetermined', 'undetermined', 'undetermined', 'undetermined',
                        str(point_ids)]
        # elif ctx.triggered[0]["prop_id"] == 'uncertainty_display_episode':
        #     if episode is not None:
        #         e_idx = e_ids[episode]
        #         p_idx = p_ids[episode]
        #         step_idx = None
        #         b_idx = b_ids[episode]
        #         data = [time, 'loss_reward', 'select episode', str(e_idx), str(p_idx), str(step_idx), str(b_idx),
        #                 str(e_idx)]
        #     else:
        #         data = [time, 'loss_reward', 'UNselect episode', 'None', 'None', 'None', 'None', 'None']
        elif ctx.triggered[0]["prop_id"] == 'loss_reward_graph.selectedData':
            if reward_loss_selectedData is not None:
                point_ids = [point['pointIndex'] for point in reward_loss_selectedData['points']]
                data = [time, 'loss_reward', 'select', 'undetermined', 'undetermined', 'undetermined', 'undetermined',
                        str(point_ids)]
        elif ctx.triggered[0]["prop_id"] == 'loss_reward_graph.hoverData':
            if reward_loss_hoverData is not None:
                e_idx = e_ids[reward_loss_hoverData['points'][0]['pointIndex']]
                p_idx = p_ids[reward_loss_hoverData['points'][0]['pointIndex']]
                b_idx = b_ids[reward_loss_hoverData['points'][0]['pointIndex']]
                data = [time, 'loss_reward', 'hover', str(e_idx), str(p_idx), 'None', str(b_idx),
                        str(reward_loss_hoverData['points'])]
        if data is not None:
            write_user_data_interactions(data, filename)
    return no_update
