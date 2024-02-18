import dash
from dash import html, dcc, Input, Output, callback, no_update, State
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from dash.exceptions import PreventUpdate

# icons for buttons and feedback segment containers
plus_icon = html.I(className="bi bi-plus", style={'font-size': '20px'})
minus_icon = html.I(className="bi bi-dash", style={'font-size': '20px'})

# icons for critique buttons:
thumbs_up_icon = html.I(className="bi bi-hand-thumbs-up", style={'font-size': '2.5vh', 'color': 'green'})
reset_icon = html.I(className="bi bi-x", style={'font-size': '2.5vh', 'color': 'grey'})
thumbs_down_icon = html.I(className="bi bi-hand-thumbs-down",
                          style={'font-size': '2.5vh', 'color': 'red'})


# subtract neighbouring slider values and return the list of differences; the differences correspond to the amounts
# of steps that belong to the step segments:
def diffs(slider_values):
    print(f'slider-values: {slider_values}')
    print(f'diffs: {[j - i for i, j in zip(slider_values[:-1], slider_values[1:])]}')
    return [j - i for i, j in zip(slider_values[:-1], slider_values[1:])]


critique_layout = html.Div([
    dcc.Graph(id='critique_graph', figure=go.Figure(),
              style={'width': '27.4vw', 'height': '8vh', 'margin-top': '1vw', 'margin-bottom': '0'},
              config={'displayModeBar': False}, clear_on_unhover=True),
    html.Div([
        # slider to manually decide which steps belong to what segment:
        dcc.RangeSlider(0, 10, value=[], id='c_slider', included=False, pushable=1,
                        allowCross=False, marks=None, tooltip={"placement": "bottom"}, className='range-slider'),
    ], style={'width': '26vw', 'padding-left': '0'}),
    html.Div([
        html.Div([
            html.Div([
                html.Label('Feedback for steps:', style={'padding': '3px', 'font-weight': 'bold'}),
                html.Label('no steps selected', id='selected_steps_label')
            ]),
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            dbc.Button([thumbs_up_icon], id='thumbs_up_btn', className='slider-button',
                                       style={'left': '50%', 'position': 'relative',
                                              'transform': 'translate(-50%, 0%)'}),
                        ]),
                        html.Label('good', style={'font-size': '1.5vh'})
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw'}),
                    html.Div([
                        html.Div([
                            dbc.Button([thumbs_down_icon], id='thumbs_down_btn', disabled=False,
                                       className='slider-button',
                                       style={'left': '50%', 'position': 'relative',
                                              'transform': 'translate(-50%, 0%)'}),
                        ]),
                        html.Label('bad', style={'font-size': '1.5vh'})
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw'}),
                    html.Div([
                        html.Div([
                            dbc.Button([reset_icon], id='reset_btn', disabled=False, className='slider-button',
                                       style={'left': '50%', 'position': 'relative',
                                              'transform': 'translate(-50%, 0%)'}),
                        ]),
                        html.Label('neutral', style={'font-size': '1.5vh'})
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw'}),
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
        ], style={'width': '16vw', 'border-right': '1px solid #bbb'}),
        html.Div([
            html.Label('Add/remove segments:', style={'padding': '3px', 'font-weight': 'bold'}),
            html.Div([
                html.Div([
                    html.Div([
                        # button to add another handle to the segmentation slider
                        dbc.Button([plus_icon], id='plus_button_critique', className='slider-button'),
                        html.Label('add', style={'font-size': '1.5vh'})
                    ], style={'padding-top': '0.5vh'}),
                    html.Div([
                        html.Div([
                            # button to remove one handle from the segmentation slider
                            dbc.Button([minus_icon], id='minus_button_critique', className='slider-button',
                                       style={'left': '50%', 'position': 'relative',
                                              'transform': 'translate(-50%, 0%)'}),
                        ]),
                        html.Label('remove', style={'font-size': '1.5vh'})
                    ], style={'padding-top': '0.5vh', 'margin-left': '1vw'}),
                ], style={'width': '5vw', 'display': 'flex', 'position': 'relative', 'margin': '0 auto'}),
            ])

        ], style={'width': '10vw'}),

    ], style={'display': 'flex', 'border-radius': '5px', 'border': '1px solid #bbb', 'width': '26vw',
              'margin-top': '1vw',

              'background-color': 'rgb(242,242,242)'}),
    # button to submit the feedback:
    dbc.Button('Submit', id='c_submit_button', style={'margin-top': '0.5vw'},
               disabled=False)
])


# this callback updates the figure property of the graph with id 'critique_graph'
@callback(Output('critique_graph', 'figure'),
          Input('feedback_tabs', 'value'),
          Input('c_slider', 'value'),
          Input('critique_graph', 'hoverData'),
          Input('selected_step_sequence', 'data'),
          Input('critique_segment_colors', 'data'),
          Input('uncertainty_display_episode', 'data'),
          State('c_slider', 'max'),
          )
def update_critique_graph(feedback_type, slider_values, hoverData, selected_step_sequence, colors, episode, maximum):
    # prevent update if feedback type is not 'critique':
    if feedback_type != 'tab-1':
        raise PreventUpdate

    if episode is None:
        fig = go.Figure()
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
        return fig

    # add 0 to slider values to enable calculation of differences:
    if 0 not in slider_values:
        slider_values.insert(0, 0)

    # add maximum to slider values to enable calculation of differences:
    if maximum not in slider_values:
        slider_values.append(maximum)

    differences = diffs(slider_values)

    # if slider_values is empty return one trace:
    x_data = [differences]

    y_data = [0]

    customdata = slider_values

    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                customdata=[f'{customdata[i]}-{customdata[i + 1]}'],
                hovertemplate='Steps %{customdata} <extra></extra>',
                # hoverlabel=dict(
                #     bgcolor='rgba(0,255,0,0)',
                #     bordercolor="rgba(24, 59, 218, 0.8)",
                # ),
                showlegend=False,
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(barmode='stack',
                      showlegend=False,
                      xaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      yaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      margin=dict(l=0, r=0, t=0, b=0),
                      plot_bgcolor='#f9f9f9',
                      # hovermode=False
                      # hoverlabel=dict(bgcolor='rgba(67, 255, 100, 0.00001)')
                      )

    if hoverData:
        # get trace index of the bar that was hovered over:
        trace_index = hoverData['points'][0]['curveNumber']

        # set x0 and x1 to the x-values of the hovered bar:
        x0 = slider_values[trace_index]
        x1 = slider_values[trace_index + 1]
        fig.add_vrect(x0=x0, x1=x1, y0=0.1, y1=0.9, line_width=2, line_color='rgb(133, 133, 133)')
    elif selected_step_sequence is not None:
        x0 = slider_values[selected_step_sequence]
        x1 = slider_values[selected_step_sequence + 1]
        fig.add_vrect(x0=x0, x1=x1, y0=0.1, y1=0.9, line_width=2, line_color='rgb(133, 133, 133)')

    return fig


@callback(
    Output('c_slider', 'min'),
    Output('c_slider', 'max'),
    Input('uncertainty_display_episode', 'data'),
    State('entropy_stepwise', 'data')
)
def update_slider_min_max(episode, entropy_stepwise):
    if episode is None:
        return None, None
    else:
        return 0, len(entropy_stepwise[episode]) - 1


@callback(
    Output('c_slider', 'value'),
    Input('plus_button_critique', 'n_clicks'),
    Input('minus_button_critique', 'n_clicks'),
    State('c_slider', 'value'),
    State('c_slider', 'max'),
    Input('uncertainty_display_episode', 'data'),
    Input('feedback_tabs', 'value')
)
def update_critique_slider_values(plus_click, minus_click, slider_values, maximum, episode, feedback_type):
    # reset slider values if feedback type is not 'critique':
    if feedback_type != 'tab-1':
        return []

    # get id of component that triggered the callback:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # reset slider values if an episode is newly selected or unselected or segmentation mode is not 'manual':
    if trigger_id == 'uncertainty_display_episode':
        return []
    # add handle (only when there are currently fewer handles than steps-1 and there is no handle at the maximum value):
    elif trigger_id == 'plus_button_critique' and len(slider_values) < maximum and maximum not in slider_values:
        slider_values.append(maximum)
        return slider_values
    # remove handle (only if there currently is at least one handle):
    elif trigger_id == 'minus_button_critique' and len(slider_values) > 0:
        list_max = max(slider_values)
        slider_values.remove(list_max)
        return slider_values
    else:
        return no_update


@callback(Output('selected_step_sequence', 'data'),
          Input('feedback_tabs', 'value'),
          Input('critique_graph', 'clickData'),
          Input('uncertainty_display_episode', 'data'),
          Input('c_slider', 'value'),
          config_prevent_initial_callbacks=True)
def update_selected_step_sequence(feedback_type, clickData, episode, slider_values):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'c-slider':
        print('selected_step_sequence trigger: c-slider')
        return None
    if trigger_id == 'uncertainty_display_episode':
        print('selected_step_sequence trigger: uncertainty_display_episode')
        return None
    if trigger_id == 'feedback_tabs':
        print('selected_step_sequence trigger: feedback_tabs')
        return None
    elif clickData:
        # get trace index of the bar that was clicked on:
        print('selected_step_sequence trigger: click-data')
        trace_index = clickData['points'][0]['curveNumber']
        return trace_index
    else:
        print('selected_step_sequence trigger: else')
        return None


@callback(Output('critique_segment_colors', 'data'),
          Input('feedback_tabs', 'value'),
          Input('c_slider', 'value'),
          Input('thumbs_up_btn', 'n_clicks'),
          Input('thumbs_down_btn', 'n_clicks'),
          Input('reset_btn', 'n_clicks'),
          Input('uncertainty_display_episode', 'data'),
          State('selected_step_sequence', 'data'),
          State('critique_segment_colors', 'data'),
          State('c_slider', 'max'),
          config_prevent_initial_callbacks=True)
def update_critique_segment_colors(feedback_type, slider_values, thumb_up_clicks, thumb_down_clicks, reset_clicks,
                                   episode, selected_step_sequence, segment_colors, maximum):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'uncertainty_display_episode' or trigger_id == 'feedback_tabs':
        return ['#c6c3c2']  # grey

    # if triggered by slider values:
    if trigger_id not in ['thumbs_up_btn', 'thumbs_down_btn', 'reset_btn', 'feedback_tabs']:
        # remove those slider values that don't add another trace, i.e. sliders' min and max values:
        if 0 in slider_values:
            slider_values.remove(0)
        if maximum in slider_values:
            slider_values.remove(maximum)

        # when a slider value has been added:
        if len(slider_values) + 1 > len(segment_colors):
            segment_colors.append('#c6c3c2')
        # when a slider value has been removed:
        elif len(slider_values) + 1 < len(segment_colors):
            segment_colors.pop()
        return segment_colors
    # if one of the feedback buttons has been clicked:
    elif selected_step_sequence is not None:
        if trigger_id == 'thumbs_up_btn':
            segment_colors[selected_step_sequence] = '#B3E5B5'  # green
            return segment_colors
        elif trigger_id == 'thumbs_down_btn':
            segment_colors[selected_step_sequence] = '#E57373'  # red
            return segment_colors
        else:
            segment_colors[selected_step_sequence] = '#c6c3c2'  # grey
            return segment_colors
    else:
        return no_update


@callback(Output('thumbs_up_btn', 'disabled'),
          Output('thumbs_down_btn', 'disabled'),
          Output('reset_btn', 'disabled'),
          Input('selected_step_sequence', 'data'),
          config_prevent_initial_callbacks=True)
def set_critique_buttons_disabled(selected_step_sequence):
    if selected_step_sequence is not None:
        return False, False, False
    else:
        return True, True, True


@callback(Output('plus_button_critique', 'disabled'),
          Output('minus_button_critique', 'disabled'),
          Input('uncertainty_display_episode', 'data'))
def update_critique_plus_minus_buttons_disabled(episode):
    if episode is None:
        return True, True
    else:
        return False, False


@callback(Output('critique_graph_hoverData', 'data'),
          Input('critique_graph', 'hoverData'),
          config_prevent_initial_callbacks=True)
def update_critique_graph_hover_data_store(hoverData):
    return hoverData


@callback(Output('critique_graph', 'clickData'),
          Input('uncertainty_display_episode', 'data'),
          Input('minus_button_critique', 'n_clicks'))
def reset_clickData(episode, minus_clicks):
    return None


@callback(Output('selected_steps_label', 'children'),
          Input('selected_step_sequence', 'data'),
          State('c_slider', 'value'),
          State('c_slider', 'max'),
          config_prevent_initial_callbacks=True)
def update_selected_steps_label(selected_step_sequence, c_slider_values, c_slider_max):
    if selected_step_sequence is None:
        return 'no steps selected'
    else:
        if 0 not in c_slider_values:
            c_slider_values.insert(0, 0)
        if c_slider_max not in c_slider_values:
            c_slider_values.append(c_slider_max)
        s1 = c_slider_values[selected_step_sequence]
        s2 = c_slider_values[selected_step_sequence + 1]
        text = f'{s1} - {s2}'
        return text


@callback(Output('c_submit_button', 'disabled'),
          Input('feedback_tabs', 'value'),
          Input('uncertainty_display_episode', 'data'),
          config_prevent_initial_callbacks=True)
def set_submit_button_disabled(feedback_type, episode):
    if feedback_type != 'tab-1':
        return True
    if episode is None:
        return True
    return False
