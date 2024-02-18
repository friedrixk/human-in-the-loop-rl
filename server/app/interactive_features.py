
from matplotlib import offsetbox
import mplcursors
from matplotlib.widgets import Button
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
from app.calculator import Calculator


class InteractiveFeatures():

  def enable_legend_picking(self, plot, lines_original, bars, legend_pos=(1,0)):
    """
    method implements legend picking. 
    It maps the visualized lines and bars to their corresponding legend items.

    :param plot: plot whose legend can be picked.
    :param lines_original: list of visualized lines.
    :param bars: list of bars displayed.
    :param legend_pos: legend's position.
    :return: tuple that contains two lists of visualized items.
    """
    leg = plot.legend(bbox_to_anchor=legend_pos, loc="lower left", fancybox=True, frameon=True)
    lined, bar_leg = {}, {}

    if leg.get_patches() and bars:
      legpatch = leg.get_patches()[0]
      legpatch.set_picker(True)
      bar_leg[legpatch] = bars[0]

    for legline, origline in zip(leg.get_lines(), lines_original):
       legline.set_picker(True)  
       lined[legline] = origline

    return (lined, bar_leg)



  def add_tooltip_standard(self, plot):
    """
    method adds a tooltip. The tooltip shows only the x and y value.
    
    :param plot: plot to which the tooltip feature is added.
    """
    c = mplcursors.cursor(plot, hover=mplcursors.HoverMode.Transient)
    @c.connect("add")
    def _(sel):
         sel.annotation.get_bbox_patch().set(height=2.5, width=2.0, boxstyle="square", fc="white", alpha=1.0)
         sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=1.0)



  def add_tooltip(self, plot, zero, nonzero, grad_norm, loss, accuracy, idx=0, episodes=[]):
    """
    method adds a tooltip. The tooltip shows the episode number, the norm of the gradient, 
    the percentage of un-/changed weights, the loss, and the accuracy value.
    Tooltip appears whenever the user hovers over a line/ bar.

    :param plot: plot to which the tooltip feature is added.
    :param zero: number of weights unchanged.
    :param nonzero: number of weights changed.
    :param grad_norm: gradient norm values that are calculated during training.
    :param loss: values of loss function that are calculated during training.
    :param accuracy: accuracy values.
    :param idx: index that indicates whether the x axis is horizontally (idx=0) or vertically (idx=1) aligned.
    :param episodes: subset of training episodes currently visible in plot.
    """
    c = mplcursors.cursor(plot, hover=mplcursors.HoverMode.Transient)
    @c.connect("add")
    def _(sel):
         if idx == 1:
            episode = episodes[round(sel.target[idx])]
         else:
            episode = round(sel.target[idx])

         (changed, unchanged) = Calculator().calculate_weight_info(episode, zero, nonzero)
         sel.annotation.get_bbox_patch().set(height=2.5, width=2.0, boxstyle="square", fc="white", alpha=1.0)
         sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=1.0)
         sel.annotation.set(text= ''.join((" episode: " + str(episode),
                                           "\n gradient norm:  "+ str(grad_norm[episode]),
                                           "\n weights changed: " + str(changed) + "%",
                                           "\n weights unchanged: " + str(unchanged) + "%",
                                           "\n loss: " + str(loss[episode]),
                                           "\n accuracy: " + str(accuracy[episode] *100) + "%"
                                           )))
         
                                


  def add_tooltip_uncertainty(self, plot, probabilities=[], actions=[], KLD=None):
    """
    method adds a tooltip. The tooltip shows the action, the action's probability, 
    the entropy of the probability distribution, and the KL divergence. If there are no probabilities, actions, and 
    no KL divergence value specified as input parameter, it only shows the step number and the KL divergence.
    Tooltip appears whenever the user hovers over a line/ bar.

    :param plot: plot to which the tooltip feature is added.
    :param probabilities: list containing probabilities for each step.
    :param actions: BabyAI action space, e.g. "left", "right", "toggle",...
    :param KLD: KL divergence.
    """
    c = mplcursors.cursor(plot, hover=mplcursors.HoverMode.Transient)
    @c.connect("add")
    def _(sel):
         sel.annotation.get_bbox_patch().set(height=2.5, width=2.0, boxstyle="square", fc="white", alpha=1.0)
         sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=1.0)

         if probabilities and actions and (KLD != None):
           content = ''.join(("action: " + str(actions[round(sel.target[0])]),
                              "\nprobability: " + str(round(probabilities[round(sel.target[0])] * 100, 2))+ "%",
                              "\nentropy: " + str(round(Calculator().calculate_entropy(probabilities), 2) ),
                              "\nKL divergence: " + str(KLD)
                              ))
          
         else:
            content = ''.join(("step: " + str(round(sel.target[0])),
                               "\nKL divergence: " + str(round(sel.target[1], 2))
                              ))

         sel.annotation.set(text= content)
 


  def annotate_percentage(self, plot, bars, scaling_factor, size, coordinates):
    """
    method annotates a percentage above each bar of a bar chart. The bar's height encodes the percentage.

    :param plot: plot that contains the annotated bar chart.
    :param bars: the bars to annotate.
    :param scaling_factor: scales the height; height is set relative to maximum y value.
    :param size: text size of annotation.
    :param coordintes: the coordinates to place the text at.
    :return: the generated annotations.
    """
    annotations = [plot.annotate(str(int(bar.get_height() / scaling_factor)) + "%",
                       (bar.get_x() + bar.get_width() / 2,
                       bar.get_height()), ha="center", va="center",
                       size=size, xytext=coordinates,
                       textcoords="offset points", color="grey")  
                   for bar in bars] 
    
    return annotations



  def add_button(self, axis, label, f):
    """
    method creates a button. It connects the button with a function. This function is invoked after the user clicks the button.

    :parm axis: button position.
    :param label: button text label.
    :param f: function that is invoked via mouse click.  
    :return: created button.
    """
    btn = Button(axis, label, color="powderblue", hovercolor="cadetblue")
    btn.on_clicked(f)
    return btn



  def add_explanation(self, plots, text):
    """
    method places a text box into an empty plot. The text instructs the user 
    to click on a data point to see some more data.

    :param plots: list of empty plots where the text appears.
    :param text: explanation text that appears in a text box on the screen.
    """
    for plot in plots:
      plot.clear()
      offsetbox = TextArea(text)
      ab = AnnotationBbox(offsetbox, xy=(0.5, 0.5), annotation_clip=False)
      plot.add_artist(ab)



  def highlight_selected_point(self, plot, point_selected, max_height):
    """
    method places a text box into a plot. The text box shows the label "selected".
    A line connects the text box with the point that the user has selected.

    :param plot: plot in which the text box appears.
    :param point_selected: the x and y coordinates of the point on which the user has clicked.
    :param max_height: the maximum value of the y axis.
    :return: the annotation box object.
    """
    offsetbox = TextArea("selected")
    offsetbox.set_zorder(10.0)
    offset = max_height - point_selected[1]
   
    ab = AnnotationBbox(offsetbox, point_selected,
                    xybox=(max_height, point_selected[1] + 20*offset),
                    xycoords='data',
                    boxcoords="offset points",
                    zorder=13.0,
                    clip_on=False,
                    alpha=1.0,
                    arrowprops=dict(arrowstyle="-", connectionstyle="angle3, angleA=0, angleB=-90"))
              
    plot.add_artist(ab)
    ab.draggable()
    return ab


    