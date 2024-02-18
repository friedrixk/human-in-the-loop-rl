import dash
from dash import html, dcc, Input, Output, callback, no_update, State, ALL
import dash_bootstrap_components as dbc
from app.parsing import SimpleParser
import numpy as np
import math
from dash.exceptions import PreventUpdate
import my_dash_component

# icons for buttons and feedback segment containers
plus_icon = html.I(className="bi bi-plus", style={'font-size': '2.5vh'})
minus_icon = html.I(className="bi bi-dash", style={'font-size': '2.5vh'})
emoji_smile = html.I(className="bi bi-emoji-smile", style={'font-size': '2vh'})
emoji_neutral = html.I(className="bi bi-emoji-neutral", style={'font-size': '2vh'})
emoji_frown = html.I(className="bi bi-emoji-frown", style={'font-size': '2vh'})
trash_icon = html.I(className="bi bi-trash", style={'font-size': '1vh'})

parser = SimpleParser()


def generate_feedback_div(segments):
    """
    Generates the div that contains the separate feedback segments. For each segment a div will be created containing
    the highlighted text of the segment and having the background color that corresponds to the nlp sentiment score of
    that particular segment. Furthermore, an emoji corresponding to the sentiment score will be displayed in the top
    left corner of the div.
    @param segments: the feedback segments
    @return: html.div
    """
    return html.Div([
        html.Div([
            html.Div([
                # emoji-div; overwritten by set_feedback_segment_container_background callback:
                html.Div([emoji_smile], id={'type': 'emoji_div', 'index': '{}'.format(seg[0])}, style={'width': '3vw'}),
                html.Div([
                    # textual representation of the steps that belong to the segment
                    html.P('Steps: 1-40',
                           style={'font-size': '1.3rem', 'margin': 0, 'position': 'relative', 'top': '50%',
                                  '-ms-transform': 'translateY(-50%)', 'transform': 'translateY(-50%)'})
                ], style={'width': '6vw'}),
                html.Div([
                    # progress bar to indicate which steps belong to the segment
                    dbc.Progress(value=40, style={'font-size': '1rem', 'height': '0.8vw', 'margin': 0,
                                                  'position': 'relative', 'top': '50%',
                                                  '-ms-transform': 'translateY(-50%)',
                                                  'transform': 'translateY(-50%)'})
                ], style={'width': '12vw'}),
                html.Div([
                    # delete button
                    dbc.Button(['Delete ', trash_icon],
                               style={'padding-left': 0, 'padding-right': 0, 'font-size': '1vh', 'height': '3vh',
                                      'width': '3.5vw', 'line-height': '1vh',
                                      'background-color': 'rgb(249, 249, 249)', 'margin': 0, 'position': 'relative',
                                      'top': '50%',
                                      '-ms-transform': 'translateY(-50%)', 'transform': 'translateY(-50%)'})
                ], style={'display': 'flex', 'height': '3vh', 'width': '4vw',
                          'padding-left': '0.25vw', 'padding-right': '0.25vw'}),
            ], style={'display': 'flex', 'height': '3vh', 'width': '25vw'}),
            html.Div([
                # Component that contains the highlighted text
                my_dash_component.MyDashComponent(id={'type': 'feedback_segment', 'index': '{}'.format(seg[0])},
                                                  value='{}'.format(seg[1]),
                                                  label='my-label',
                                                  highlight=[  # overwritten by highlight_feedback_segments callback
                                                      {
                                                          'highlight': 'bad',
                                                          'className': 'red'
                                                      },
                                                      {
                                                          'highlight': 'good',
                                                          'className': 'green'
                                                      }
                                                  ]),
            ], style={'padding': '6px 10px',
                      'width': '24vw', 'text-align': 'left', 'border': '1px solid #d1d1d1', 'height': '8vh',
                      'border-radius': '4px', 'background-color': '#fff', 'margin': '1vh 0', 'overflow-y': 'scroll',
                      'overflow-x': 'hidden'}),
        ], id={'type': 'feedback_segment_container', 'index': '{}'.format(seg[0])},
            style={'height': '14.5vh', 'width': '25vw', 'padding-left': '0.5vw', 'padding-top': '0.5vw',
                   'padding-right': '0.5vw', 'margin-top': '0.8vh',
                   'border-radius': '6px'}) for seg in segments  # generate for each feedback segment
    ], id={'type': 'feedback_segment_wrapper', 'index': 0},
        style={'margin-top': '0.5vh', 'height': '33vh', 'overflow-y': 'scroll'})


# fk: taken from Felix HuBaFeedRL feedbackbox.py:
def sentiment_analysis(text: list) -> (list, list):
    processed = parser.nlp(text)
    evaluation = processed._.blob.sentiment_assessments.assessments
    keywords = [x[0] for x in evaluation]
    polarity = [x[1] for x in evaluation]
    return keywords, polarity


def sentiment_to_color(polarities: list) -> list:
    colors = []
    for pol in polarities:
        if pol > 0.01:
            colors.append('green')
        elif pol < -0.01:
            colors.append('red')
        else:
            colors.append('neutral')
    return colors


def sentiment_to_emoji(polarities: list) -> list:
    emojis = []
    for pol in polarities:
        if pol > 0.01:
            emojis.append(emoji_smile)
        elif pol < -0.01:
            emojis.append(emoji_frown)
        else:
            emojis.append(emoji_neutral)
    return emojis


# fk: partly adapted from Felix HuBaFeedRL feedbackbox.py:
def sentiment_to_highlight(input_text: str) -> dict:
    keywords_old, polarity = sentiment_analysis(input_text)
    keywords_new = []
    highlight = []
    # keywords_new = ([' '.join(k) for k in keywords_old])
    for i, k in enumerate(keywords_old):
        if k[len(k) - 1] == '!':
            k = k[:-1]
            k[len(k) - 1] = k[len(k) - 1] + '!'
            keywords_old[i] = k
    keywords_new = ([' '.join(k) for k in keywords_old])
    sentiment = sorted(list(zip(keywords_new, polarity)), key=lambda a: len(a[0]), reverse=True)
    res = list(zip(*sentiment))
    if res:
        keywords_new = res[0]
        polarity = res[1]
    for i, k in enumerate(keywords_new):
        if polarity[i] > 0:
            highlight.append({
                'highlight': k,
                'className': 'green'
            })
        elif polarity[i] < 0:
            highlight.append({
                'highlight': k,
                'className': 'red'
            })
        else:
            highlight.append({
                'highlight': k,
                'className': 'neutral'
            })
    return highlight


natural_language_feedback_layout = html.Div([
    dcc.Store(id='nl_segment_colors'),
    html.Div([
        # input for feedback:
        my_dash_component.MyDashComponent(id='input',
                                          value='Type feedback ...',
                                          label='my-label',
                                          highlight=[
                                              {
                                                  'highlight': 'bad',
                                                  'className': 'red'
                                              },
                                              {
                                                  'highlight': 'good',
                                                  'className': 'green'
                                              }
                                          ]),
    ], style={'padding-top': '6px', 'padding-bottom': '6px', 'padding-left': '10px', 'padding-right': '10px',
              'width': '25vw', 'text-align': 'left', 'border': '1px solid #d1d1d1', 'height': '20vh',
              'border-radius': '4px', 'background-color': '#fff', 'margin': '1vh 0', 'overflow-y': 'scroll',
              'overflow-x': 'hidden'}),
    html.Div([
        html.Div([
            html.Label('Segmentation mode:',
                       style={'margin': 0, 'position': 'relative', 'top': '50%', '-ms-transform': 'translateY(-50%)',
                              'transform': 'translateY(-50%)'})
        ]),
        # dropdown for segmentation mode selection:
        dcc.Dropdown(['simple', 'entropy', 'manual'], value='simple', id='segmentation_mode_dropdown',
                     style={'textAlign': 'left', 'width': '10vw', 'margin-left': '5px'}),
        html.Div([
            # button to trigger segmentation of feedback:
            dbc.Button('Segment', id='segment_button', disabled=True,
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'height': '36px',
                              'margin-left': '1vw'})
        ]),
    ], style={'display': 'flex', 'margin-bottom': '5px'}),
    html.Div([
        html.Div([
            # slider to manually decide which steps belong to what segment:
            dcc.RangeSlider(0, 1, id='nl_slider', included=False, pushable=1, allowCross=False,
                            marks=None, tooltip={"placement": "bottom"})
        ], style={'width': '20vw'}),
        html.Div([
            # button to add another handle to the segmentation slider
            dbc.Button([plus_icon], id='plus_button',
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'margin-left': '0.5vw',
                              'height': '36px'}),
            # button to remove one handle from the segmentation slider
            dbc.Button([minus_icon], id='minus_button',
                       style={'padding-left': '0.5vw', 'padding-right': '0.5vw', 'margin-left': '3px',
                              'height': '36px'}),
        ], id='plus_minus_buttons')
    ], id='slider_div', style={'width': '26vw', 'display': 'flex'}),
    html.Div([
    ], id='segments_div'),
    # button to submit the feedback:
    dbc.Button('Submit', id='nl_submit_button', style={'margin-top': '0.5vw'},
               disabled=True)
])


@callback(
    Output('nl_slider', 'min'),
    Output('nl_slider', 'max'),
    Input('uncertainty_display_episode', 'data'),
    State('entropy_stepwise', 'data')
)
def update_slider_min_max(episode, entropy_stepwise):
    if episode is None:
        return None, None
    else:
        return 0, len(entropy_stepwise[episode]) - 1


@callback(
    Output('nl_slider', 'value'),
    Input('plus_button', 'n_clicks'),
    Input('minus_button', 'n_clicks'),
    Input('feedback_tabs', 'value'),
    State('nl_slider', 'value'),
    State('nl_slider', 'max'),
    Input('uncertainty_display_episode', 'data'),
    Input('segmentation_mode_dropdown', 'value')
)
def update_slider_values(plus_click, minus_click, feedback_type, slider_values, maximum, episode, segmentation_mode):
    # reset slider values if feedback type is not 'natural language':
    if feedback_type != 'tab-2':
        print('option0')
        return []

    # get id of component that triggered the callback:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == 'segmentation_mode_dropdown':
        return []

    # reset slider values if an episode is (un)selected or segmentation mode not 'manual' or feedback_type is critique:
    if trigger_id == 'uncertainty_display_episode' or segmentation_mode != 'manual' or feedback_type == 'tab-1':
        print('option1')
        return []
    # add handle (only when there are currently fewer handles than steps-1):
    elif trigger_id == 'plus_button' and len(slider_values) < maximum:
        print('option2')
        print(f'maximum is {maximum}, slider_values is {slider_values}')
        slider_values.append(maximum)
        return slider_values
    # remove handle (only if there currently is at least one handle):
    elif trigger_id == 'minus_button' and len(slider_values) > 0:
        print('option4')
        list_max = max(slider_values)
        slider_values.remove(list_max)
        return slider_values
    else:
        print('option5')
        return no_update


@callback(
    Output('plus_button', 'disabled'),
    Output('minus_button', 'disabled'),
    Input('segmentation_mode_dropdown', 'value'),
    Input('uncertainty_display_episode', 'data')
)
def set_plus_minus_buttons_disabled(segmentation_mode, episode):
    if episode is None:
        return True, True
    elif segmentation_mode == 'manual':
        return False, False
    else:
        return True, True


@callback(
    Output('nl_slider', 'disabled'),
    Input('segmentation_mode_dropdown', 'value'),
    Input('uncertainty_display_episode', 'data')
)
def update_slider_disable(segmentation_mode, episode):
    if episode is None:
        return True
    elif segmentation_mode == 'manual':
        return False
    else:
        return True


@callback(
    Output({'type': 'feedback_segment_wrapper', 'index': ALL}, 'children'),
    Output('segment_button_workaround', 'data'),
    Input('segment_button', 'n_clicks'),
    State({'type': 'feedback_segment_wrapper', 'index': ALL}, 'children'),
    config_prevent_initial_callbacks=True
)
def increase_segment_button_workaround(click, children):
    """
    When segment button is clicked, set the children of all components with id-type 'feedback_segment_wrapper' to None.
    This is a workaround for programmatically setting the value of my_dash_component components since my_dash_component
    does not allow this. Directly after this callback, the children of 'feedback_segment_wrapper' will be
    (re-)instantiated with updated values by the 'update_segments_div' callback which will be triggered by the changed
    value of 'segment_button_workaround'
    @param click: the number of times the segment button has been clicked
    @param children: the children of all 'feedback_segment_wrapper' components
    @return: None for each 'feedback_segment_wrapper' and current value of n_clicks property of the segment button
    """
    return [None for i in children], click


@callback(
    Output('segments_div', 'children'),
    State('nl_slider', 'value'),
    State('nl_slider', 'min'),
    State('nl_slider', 'max'),
    State('segmentation_mode_dropdown', 'value'),
    State('input', 'value'),
    Input('segment_button_workaround', 'data'),
    Input('uncertainty_display_episode', 'data'),
    config_prevent_initial_callbacks=True
)
def update_segments_div(slider_values, slider_min, slider_max, segmentation_mode, input_text, n_clicks, episode):
    """
    Updates the segments_div to show each segment in a separate text input field of type my_dash_component. It will be
    triggered either if the selected episode (uncertainty_display_episode) changes or if segment_button is clicked
    :param slider_values:
    :param segmentation_mode:
    :param n_clicks:
    :return:
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # if selected episode changes (None or other episode), segments_div shall be reset to None
    if trigger_id == 'uncertainty_display_episode':
        return None

    # if segmentation_mode is not selected, segments_div shall be (re)set to be empty:
    if segmentation_mode is None:
        return None

    # if segmentation mode is manual:
    if segmentation_mode == 'manual':
        # remove min and max from values if they are contained (they shall not be taken into account for segmentation):
        if slider_min in slider_values:
            slider_values.remove(slider_min)
        if slider_max in slider_values:
            slider_values.remove(slider_max)
        # make segmentation of sentences:
        segments = [sent for sent in enumerate(parser.nlp(input_text).sents)]
        # only take into account the first len(slider_values)+1 sentences:
        segments = segments[:len(slider_values) + 1]
        return generate_feedback_div(segments)

    # if segmentation mode is simple:
    if segmentation_mode == 'simple':
        segments = [sent for sent in enumerate(parser.nlp(input_text).sents)]
        return generate_feedback_div(segments)
    else:
        return no_update


@callback(
    Output('input', 'highlight'),
    Input('input', 'value'),
    config_prevent_initial_callbacks=True
)
def highlight_input_text(input_text):
    return sentiment_to_highlight(input_text)


@callback(Output({'type': 'feedback_segment', 'index': ALL}, 'highlight'),
          Input({'type': 'feedback_segment', 'index': ALL}, 'value'))
def highlight_feedback_segments(feedback_segments):
    highlights = []
    for fs in feedback_segments:
        highlights.append(sentiment_to_highlight(fs))
    return highlights


@callback(Output({'type': 'feedback_segment_container', 'index': ALL}, 'className'),
          Output({'type': 'emoji_div', 'index': ALL}, 'children'),
          Output('nl_segment_colors', 'data'),
          Input({'type': 'feedback_segment', 'index': ALL}, 'value'),
          config_prevent_initial_callbacks=True)
def set_feedback_segment_container_background(feedback_segments):
    sentiments = []
    for fs in feedback_segments:
        polarity = sentiment_analysis(fs)[1]
        if not polarity:
            sentiments.append(0)
        else:
            sentiments.append(np.mean(polarity))
    classes = sentiment_to_color(sentiments)
    emojis = sentiment_to_emoji(sentiments)
    colors = []
    for c in classes:
        if c == 'green':
            colors.append('#B3E5B5')
        elif c == 'red':
            colors.append('#E57373')
        else:
            colors.append('#c6c3c2')
    return classes, emojis, colors


@callback(Output('segment_button', 'disabled'),
          Input('uncertainty_display_episode', 'data'),
          config_prevent_initial_callbacks=True)
def set_segment_button_disabled(episode):
    if episode is None:
        return True
    return False


@callback(Output('nl_submit_button', 'disabled'),
          Input('feedback_tabs', 'value'),
          Input('segments_div', 'children'),
          Input('uncertainty_display_episode', 'data'),
          config_prevent_initial_callbacks=True)
def set_submit_button_disabled(feedback_type, children, episode):
    # prevent update if feedback type is not 'natural language' (because 'segments_div' is non-existent then):
    if feedback_type != 'tab-2':
        raise PreventUpdate
    if children is None or episode is None:
        return True
    return False
