from __future__ import annotations
from argparse import ArgumentParser
from math import ceil

import matplotlib
from matplotlib import artist
from matplotlib import patches
from matplotlib import offsetbox
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import matplotlib.image as mpimg
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

import numpy as np
from IPython import display
from matplotlib.widgets import Button
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import RangeSlider
from app.filter import Filter
from app.utility_vis import UtilityVis
from app.interactive_features import InteractiveFeatures
from app.file_handler import FileHandler
from app.calculator import Calculator
from app.visualizations import PlotGrid
from app.glyph_creator import Glyph
from app.sampler import Sampler

parser = ArgumentParser()
parser.add_argument("--env", help="name of the environment to be run (REQUIRED)")
parser.add_argument("--file_seed_list", required=True,
                    help="specify path to file that stores the seeds used in demo generation")
parser.add_argument("--file_path", required=True, help="specify path to file that stores the visualization data")
parser.add_argument("--save_path", required=True, help="specify path to save the visualizations")


class GradNormVis():

    def __init__(self):

        matplotlib.style.use("seaborn-dark")

        # figure 1 shows the main plot and the zoom area
        self.fig = plt.figure(1, figsize=(9.5, 9.5), dpi=120)

        # figure 2
        self.fig2 = plt.figure(2)
        plt.figure(2)
        plt.close()

        plt.figure(1)

        # grid contains 3 plots, main plot (plot1) and distance to target plot (plot2), and return values plot (plot3)
        plt.figure(1)
        self.plot1 = plt.subplot2grid((310, 300), (3, 0), rowspan=143, colspan=300)
        self.plot2 = plt.subplot2grid((310, 300), (200, 0), rowspan=135, colspan=136)
        self.plot3 = plt.subplot2grid((310, 300), (200, 150), rowspan=135, colspan=136)
        # self.plot4 = plt.subplot2grid((310,300), (200, 290), rowspan=135, colspan=20)
        plt.tight_layout()

        # workaround matplotlib/ipympl #290
        if UtilityVis().type_of_script() == 'jupyter':
            display.display(self.fig.canvas)
            self.fig.canvas._handle_message(self.fig.canvas, {'type': 'send_image_mode'}, [])
            self.fig.canvas._handle_message(self.fig.canvas, {'type': 'refresh'}, [])
            self.fig.canvas._handle_message(self.fig.canvas, {'type': 'initialized'}, [])
            self.fig.canvas._handle_message(self.fig.canvas, {'type': 'draw'}, [])

    def set_plotdata(self, env_name, grad_norm, zero, nonzero, loss, actions,
                     actions_true, seeds, accuracy, all_episodes, episodes_fail, episodes_success):
        """
        method takes as input parameters the data to visualize and initializes the class variables.

        :param env_name: name of trained level.
        :param grad_norm: gradient norm values that are calculated during training.
        :param zero: number of weights unchanged.
        :param nonzero: number of weights changed.
        :param loss: values of loss function that are calculated during training.
        :param actions: action sequences used to calculate trajectory glyph.
        :param actions_true: true actions specified in trainings demo.
        :param seeds: action sequences used to calculate trajectory glyph.
        :param accuracy: accuracy values.
        """
        self.env = env_name
        self.grad_norm = grad_norm
        self.zero = zero
        self.nonzero = nonzero
        self.loss = loss
        self.actions = actions
        self.actions_true = actions_true
        self.seeds = seeds
        self.accuracy = accuracy
        self.selected_traj = []
        self.color_blind = False
        self.current_xrange = np.arange(len(self.grad_norm))
        self.current_episodes, self.all_episodes = all_episodes, all_episodes
        self.episodes_fail = episodes_fail
        self.episodes_success = episodes_success

    def toggle_color_blind(self, _):
        """
        method changes value of class variable color_blind. If it is true/false, the method sets it to false/true.

        :param _: button click event.
        """
        self.color_blind = not self.color_blind

    def on_pick(self, event):
        """
        method implements legend picking. The user can select which lines are visible.

        :param event: picking event.
        """
        # main plot legend is picked
        if event.artist in lined:
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)

            if origline == grad_line:
                grad_points.set_visible(visible)
            if origline == loss_line:
                loss_points.set_visible(visible)

            legline.set_alpha(1.0)

        if event.artist in bar_leg:
            for i, bar in enumerate(bars):
                visible = not bar.get_visible()
                bar.set_visible(visible)
                visible_ann = not annotations[i].get_visible()
                annotations[i].set_visible(visible_ann)
                event.artist.set_alpha(0.3)

            for bar_grey in bars_grey:
                visible = not bar_grey.get_visible()
                bar_grey.set_visible(visible)

        self.fig.canvas.draw()

    def on_click(self, event):
        """
        method shows trajectory glyph for clicked gradient norm value.
        If the trajectory of selected point is already visible, it removes the trajectory glyph.
        The method places/ removes the glyphs only if the user uses a double click.

        :param event: double click event.
        """
        if event.dblclick and self.plot1.in_axes(event):
            threshold = 0.2
            mouse_x = round(event.xdata)
            # mouse_y = round(event.ydata)

            if (event.ydata >= self.grad_norm[mouse_x] - threshold
                    and event.ydata <= self.grad_norm[mouse_x] + threshold
                    and grad_line.get_visible()):

                # glyph already displayed
                if mouse_x in [traj[0] for traj in self.selected_traj]:
                    glyph.remove_glyph(mouse_x, "traj")
                    self.selected_traj.remove((mouse_x, self.grad_norm[mouse_x]))

                else:
                    glyph.create_trajectory_glyph(trajectory_glyph_path, self.env, self.seeds[mouse_x],
                                                  self.actions[mouse_x], self.actions_true[mouse_x], mouse_x)
                    # glyph.create_pie_glyph(pie_glyph_path, self.accuracy[mouse_x], mouse_x)
                    glyph.place_glyph(self.plot1, trajectory_glyph_path, "traj", mouse_x, self.grad_norm[mouse_x])
                    # glyph.place_glyph(self.plot2, pie_glyph_path, "pie", mouse_x, mouse_y)

                    plt.figure(1)
                    self.selected_traj.append((mouse_x, self.grad_norm[mouse_x]))

                self.synchronize_plots(self.plot2, self.plot3)
                self.fig.canvas.draw()
            else:
                print("no trajectory available for clicked point")

    def brush(self, eclick, erelease):
        """
        method sets the xlim and ylim of the zoom area. It invokes the update_zoom method.

        :param eclick: mouse button click event.
        :param erelease: mouse button release event.
        """

        # (xmin, xmax, ymin, ymax) values for rectangle selector box
        extent = rect_selector.extents

        # zoom the selected brush rectangle
        # set x axis range as xmin to xmax
        self.plot1.set_xlim(extent[0], extent[1])

        # set ylim range as ymin to ymax
        self.plot1.set_ylim(extent[2], extent[3])

        # update data to display
        self.update_zoom(self.plot1, extent[0], extent[1])

    def filter_episodes(self, event):
        """
        If the user clicks on a filter button, the click invokes this method.
        It selects the data of failed episodes or of successful episodes.
        The method triggers that the filtered data appear on the display.
        Moreover, the method changes the colors of the buttons.
        The selected button is blue. The other buttons appear grey.

        :param event: button click event.
        """
        if axes_all_eps.in_axes(event):
            self.current_episodes = self.all_episodes
            btn_all_eps.color = "powderblue"
            btn_fail_eps.color = "lightgrey"
            btn_success_eps.color = "lightgrey"

        elif axes_success_eps.in_axes(event):
            self.current_episodes = self.episodes_success
            btn_success_eps.color = "powderblue"
            btn_all_eps.color = "lightgrey"
            btn_fail_eps.color = "lightgrey"

        elif axes_fail_eps.in_axes(event):
            self.current_episodes = self.episodes_fail
            btn_fail_eps.color = "powderblue"
            btn_all_eps.color = "lightgrey"
            btn_success_eps.color = "lightgrey"

        episodes_filtered = [e for e in self.current_episodes if
                             e >= self.current_xrange[0] and e <= self.current_xrange[-1]]

        if episodes_filtered:
            # select new data
            accuracy_filtered = [self.accuracy[i] for i in episodes_filtered]
            episodes_grey = [i for i in self.current_xrange if i not in episodes_filtered]
            grad_norm_grey = [self.grad_norm[i] for i in episodes_grey if i < len(self.grad_norm) - 1]
            loss_grey = [self.loss[i] for i in episodes_grey if i < len(self.loss) - 1]
            accuracy_grey = [self.accuracy[i] for i in episodes_grey]

            y_max = max(max(self.grad_norm), max(self.loss))

            # display new data
            self.create_grad_norm(self.plot1, self.all_episodes, self.current_xrange, y_max, self.grad_norm,
                                  self.loss, episodes_grey, grad_norm_grey, loss_grey)
            self.create_accuracy_bars(self.plot1, episodes_filtered, y_max, accuracy_filtered, episodes_grey,
                                      accuracy_grey)

        else:
            InteractiveFeatures().add_explanation([self.plot1, self.plot2, self.plot3],
                                                  "No episode fulfills the filter criterion.")

    def create_grad_norm(self, plot, episodes, x_range, y_max, grad_norm, loss,
                         x_grey, grad_norm_grey, loss_grey):
        """
        method creates the main plot. This line chart shows the gradient norm and the loss values.
        The visualization includes the features legend picking, a tooltip, and brushing.

        :param plot: subplot in which line chart appears.
        :param episodes: list of episode identifier.
        :param x_range: range of values shown on x axis.
        :param y_max: maximum value of y axis.
        :param grad_norm: list contains the gradient norm for each episode.
        :param loss: list contains the results of loss function for each episode.
        :param x_grey: episodes that do not fulfill the filter criteria (displayed in grey).
        :param grad_norm_grey: list contains the gradient norm for each episode that does not fulfill the filter criterion.
        :param loss_grey: list contains the results of loss function for each episode that does not fulfill the filter criterion.
        """
        plot.clear()
        plot.grid()
        plot.set_xlim(x_range[0] - 0.5, x_range[-1])
        plot.set_ylim(-0.5, y_max + 1)
        plot.set_ylabel("gradient norm and loss")

        # manually set second y axis
        # creating second y axis with plot.twinx() leads to dependencies that hamper brush feature
        plot.text(plot.get_xlim()[1] - 0.3, plot.get_ylim()[1] + 0.5, "accuracy")
        plot.text(plot.get_xlim()[1], plot.get_ylim()[0], "0%")
        plot.text(plot.get_xlim()[1], plot.get_ylim()[1] * 0.5, "50%")
        plot.text(plot.get_xlim()[1], plot.get_ylim()[1], "100%")

        global grad_line, loss_line, grad_points, loss_points, rect_selector

        # plot data
        grad_line, = plot.plot(episodes, grad_norm, lw=2, marker=".", label="gradient norm", c="tab:blue")
        loss_line, = plot.plot(episodes, loss, lw=2, marker=".", label="loss", c="tab:cyan")

        # color points that do not fulfill the filter criteria in grey
        grad_points = plot.scatter(x_grey[0:len(grad_norm_grey)], grad_norm_grey, c="grey", marker=".", zorder=10.0)
        loss_points = plot.scatter(x_grey[0:len(loss_grey)], loss_grey, c="grey", marker=".", zorder=10.0)

        # place glyphs
        eps = [e for e in x_range if e not in x_grey]
        self.show_glyph(eps)

        # tooltip
        InteractiveFeatures().add_tooltip(plot, self.zero, self.nonzero, self.grad_norm, self.loss, self.accuracy)

        # brushing via right mouse button
        rect_selector = RectangleSelector(plot, self.brush, button=[3])

    def create_accuracy_bars(self, plot, x_range, y_max, accuracy, episodes_grey, accuracy_grey):
        """
        method creates a bar chart.

        :param plot: plot shows the bars.
        :param x_range: x_range list contains the x values of the bars.
        :param y_max: maximum y value of plot.
        :param accuracy: values that are encoded in the bar's height.
        :param episodes_grey: list contains episodes that do not fulfill the filter criteria (displayed in grey).
        :param accuracy_grey: accuracy values belonging to episodes in episodes_grey.
        """

        bar_width = 0.5

        # bar height encodes percentage
        # height is relative to maximum y value of line chart
        scaling_factor = y_max / 100
        y = [scaling_factor * (i * 100) for i in accuracy]
        y_grey = [scaling_factor * (i * 100) for i in accuracy_grey]

        global bars, bars_grey
        bars = plot.bar(x_range, y, bar_width, color="yellowgreen", alpha=0.4, label="accuracy", zorder=1)
        bars_grey = plot.bar(episodes_grey, y_grey, bar_width, color="grey", alpha=0.4, zorder=1)

        # add percentage above each bar
        global annotations
        annotations = InteractiveFeatures().annotate_percentage(plot, bars, scaling_factor, 5, (0, 8))

        # add legend picking
        global lined, bar_leg
        (lined, bar_leg) = InteractiveFeatures().enable_legend_picking(plot, [grad_line, loss_line], bars, (1.01, 0))

    def show_glyph(self, episodes):
        """
        method places glyphs at fixed positions. There are five glyphs shown at any point in time.

        :param episodes: x values of main plot currently selected via slider.
        """
        if len(episodes) < 1:
            return

        self.selected_traj.clear()

        # step size depends on x range, ensures that five glyphs are shown independent of x range
        step = ceil((episodes[-1] - episodes[0]) / 5)

        x_positions = []
        i = 0
        if step != 0:
            while i < len(episodes) - 1:
                x_positions.append(episodes[i])
                i += step

        for x_pos in x_positions:
            glyph.create_trajectory_glyph(trajectory_glyph_path, self.env, self.seeds[x_pos],
                                          self.actions[x_pos], self.actions_true[x_pos], x_pos)
            glyph.place_glyph(self.plot1, trajectory_glyph_path, "traj", x_pos, self.grad_norm[x_pos])
            self.selected_traj.append((x_pos, self.grad_norm[x_pos]))

        # distances and return values for displayed trajectory glyphs
        self.synchronize_plots(self.plot2, self.plot3)

    def remove_all_glyphs(self, _):
        """
        method removes all glyphs from the display after the user clicked on the button "remove all glyphs".

        :param event: button click event.
        """
        glyph.remove_all_glyphs("all")
        self.selected_traj.clear()
        InteractiveFeatures().add_explanation([self.plot2, self.plot3],
                                              "Click on a point of the line above \nto see the corresponding data")
        self.fig.canvas.draw()

    def synchronize_plots(self, plot2, plot3):
        """
        method synchronizes the distance-to-target plot and the actions-per-episode plot.
        If the user selects a new trajectory glyph, the plots include the distances and the actions of the new trajectory.
        If the user removes a glyph, the distances and the actions disappear.

        :param plot2: visualization that shows for each step of a trajectory whether the distance to the target is in-/decreased.
        :param plot3: visualization that shows the actions for each trajectory.
        """
        if self.selected_traj:
            # collect new data
            episodes = [traj[0] for traj in self.selected_traj]
            np.unique(episodes).tolist()
            episodes.sort()

            # display at most 15 bars to avoid visual clutter
            if len(episodes) > 15:
                episodes = Sampler().systematic_random_sampling(episodes, 15)

            seed_list = [self.seeds[i] for i in episodes]
            actions = [self.actions[i] for i in episodes]
            # return_values = [return_list[i] for i in episodes]

            # visualize selected data
            vis.create_distance_plot(plot2, None, seed_list, actions, False, (0, -0.3), self.color_blind)
            vis.create_stacked_horizontal_barchart(plot3, None, seed_list, actions, False, (0, -0.3), 3)
            # vis.create_return_plot(self.plot4, return_values)

            # update axes and add tooltip
            for plot in [plot2, plot3]:
                plot.set_yticks(np.arange(len(episodes)))
                plot.set_yticklabels(episodes)
                plot.invert_yaxis()

                InteractiveFeatures().add_tooltip(plot, self.zero, self.nonzero, self.grad_norm,
                                                  self.loss, self.accuracy, 1, episodes)


        else:
            InteractiveFeatures().add_explanation([plot2, plot3],
                                                  "Click on a point of the line above \nto see the corresponding data")

    def update(self, val):
        """
        method updates the main plot whenever the user selects another range in the slider.

        :param val: new x range selected by the user.
        """
        # collect new data
        self.current_xrange = np.arange(int(val[0]), int(val[1]), 1)
        episodes = [x_i for x_i in self.all_episodes if x_i >= val[0] and x_i <= val[1]]
        episodes_green = [x_i for x_i in self.current_episodes if x_i >= val[0] and x_i <= val[1]]
        episodes_grey = [x_i for x_i in self.all_episodes if x_i not in episodes_green]

        y_max = max(max(self.grad_norm), max(self.loss))

        grad_norm_selection = [self.grad_norm[i] for i in episodes]
        loss_selection = [self.loss[i] for i in episodes]
        accuracy_selection = [self.accuracy[i] for i in episodes_green]

        grad_norm_grey = [self.grad_norm[i] for i in episodes_grey]
        loss_grey = [self.loss[i] for i in episodes_grey]
        accuracy_grey = [self.accuracy[i] for i in episodes_grey]

        # visualize selected data
        self.create_grad_norm(self.plot1, episodes, self.current_xrange, y_max, grad_norm_selection, loss_selection,
                              episodes_grey, grad_norm_grey, loss_grey)

        # remove old bars to avoid overplotting/ visual clutter
        for bar in bars:
            bar.remove()

        for bar_grey in bars_grey:
            bar_grey.remove()

        self.create_accuracy_bars(self.plot1, episodes_green, y_max, accuracy_selection, episodes_grey, accuracy_grey)

        # redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def update_zoom(self, plot, xmin, xmax):
        """
        method that displays the data of brushed area in zoom area.

        :param plot: zoom area plot.
        :param xmin: left x value of brushed area and minimum x value of zoom area.
        :param xmax: right x value of brushed area and maximum x value of zoom area.
        """
        # reset
        self.selected_traj.clear()
        slider_axes.set_visible(not slider_axes.get_visible())
        glyph.init_display_list()
        InteractiveFeatures().add_explanation([self.plot2, self.plot3],
                                              "Click on a point of the line above \nto see the corresponding data")

        xmin, xmax = int(xmin), int(xmax)
        episodes = [x_i for x_i in self.all_episodes if x_i >= xmin and x_i <= xmax]
        episodes_green = [x_i for x_i in self.current_episodes if x_i >= xmin and x_i <= xmax]
        episodes_grey = [x_i for x_i in self.all_episodes if x_i not in episodes_green]

        # select data of brush area
        grad_norm_zoom = [self.grad_norm[i] for i in episodes]
        loss_zoom = [self.loss[i] for i in episodes]
        accuracy_zoom = [self.accuracy[i] for i in episodes_green]

        grad_norm_grey = [self.grad_norm[i] for i in episodes_grey]
        loss_grey = [self.loss[i] for i in episodes_grey]
        accuracy_grey = [self.accuracy[i] for i in episodes_grey]

        # rescale x and y axis
        self.current_xrange = np.arange(xmin, xmax + 1, 1)

        if grad_norm_zoom and loss_zoom:
            y_max = max(max(grad_norm_zoom), max(loss_zoom))
        else:
            y_max = 2

        # plot data of brush area
        previous_visiblity_grad_line = grad_line.get_visible()
        previous_visiblity_loss_line = loss_line.get_visible()
        previous_visiblity_bar = bars[0].get_visible()

        self.create_grad_norm(self.plot1, episodes, self.current_xrange, y_max, grad_norm_zoom, loss_zoom,
                              episodes_grey, grad_norm_grey, loss_grey)
        self.create_accuracy_bars(plot, episodes_green, y_max, accuracy_zoom, episodes_grey, accuracy_grey)

        # show only the items that were visible before the brush
        grad_line.set_visible(previous_visiblity_grad_line)
        grad_points.set_visible(previous_visiblity_grad_line)
        loss_line.set_visible(previous_visiblity_loss_line)
        loss_points.set_visible(previous_visiblity_loss_line)

        for i, bar in enumerate(bars):
            bar.set_visible(previous_visiblity_bar)
            annotations[i].set_visible(previous_visiblity_bar)

        for bar_grey in bars_grey:
            bar_grey.set_visible(previous_visiblity_bar)

        if not previous_visiblity_grad_line:
            self.remove_all_glyphs("all")

    def create_grid(self, file_handler):
        """
        method arranges three plots in a window. It adds a range slider and buttons.

        :param file_handler: file handler saves the figure in a file.
        """
        plt.figure(1)

        x_range = np.arange(max(len(self.grad_norm), len(self.loss)))
        y_max = max(max(self.grad_norm), max(self.loss))

        self.create_grad_norm(self.plot1, x_range, x_range, y_max, self.grad_norm, self.loss, [], [], [])
        self.create_accuracy_bars(self.plot1, x_range, y_max, self.accuracy, [], [])
        self.synchronize_plots(self.plot2, self.plot3)

        # add slider
        global slider_axes, slider
        slider_axes = plt.axes([0.22, 0.45, 0.65, 0.02])
        slider = RangeSlider(slider_axes, "Select episodes", 0, len(self.grad_norm), valinit=(0, len(self.grad_norm)))
        slider.on_changed(self.update)

        # add buttons
        global btn_save, btn_import, btn_remove_glyph, btn_slider, btn_color_blind, btn_fail_eps, btn_success_eps, btn_all_eps
        global axes_fail_eps, axes_success_eps, axes_all_eps

        # axes to place buttons
        axes_save = plt.axes([0.93, 0.000002, 0.065, 0.06])
        axes_remove = plt.axes([0.93, 0.62, 0.065, 0.06])
        axes_show_slider = plt.axes([0.93, 0.44, 0.065, 0.06])
        axes_all_eps = plt.axes([0.93, 0.69, 0.065, 0.06])
        axes_fail_eps = plt.axes([0.93, 0.76, 0.065, 0.06])
        axes_success_eps = plt.axes([0.93, 0.83, 0.065, 0.06])
        axes_color_blind = plt.axes([0.25, 0.03, 0.08, 0.064])

        icon_save = plt.imread(
            "./save_button.jpg")  # 221028 fred: changed to "scripts/save_button" to "./save_button
        btn_save = Button(axes_save, label="", color="white", hovercolor="cadetblue", image=icon_save)
        btn_save.on_clicked(file_handler.save)

        btn_remove_glyph = Button(axes_remove, "remove all glyphs", color="powderblue", hovercolor="cadetblue")
        btn_remove_glyph.on_clicked(self.remove_all_glyphs)

        btn_slider = Button(axes_show_slider, "show slider", color="powderblue", hovercolor="cadetblue")
        btn_slider.on_clicked(lambda _: slider_axes.set_visible(not slider_axes.get_visible()))

        btn_all_eps = Button(axes_all_eps, "show all\n episodes", color="powderblue", hovercolor="cadetblue")
        btn_all_eps.on_clicked(self.filter_episodes)

        btn_fail_eps = Button(axes_fail_eps, "show fail\n episodes", color="powderblue", hovercolor="cadetblue")
        btn_fail_eps.on_clicked(self.filter_episodes)

        btn_success_eps = Button(axes_success_eps, "show successful\n episodes", color="powderblue",
                                 hovercolor="cadetblue")
        btn_success_eps.on_clicked(self.filter_episodes)

        btn_color_blind = Button(axes_color_blind, "change colors", color="powderblue", hovercolor="cadetblue")
        btn_color_blind.on_clicked(self.toggle_color_blind)

        if UtilityVis().type_of_script() == 'jupyter':
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

            self.fig.canvas.mpl_connect("pick_event", self.on_pick)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)

def main(file_path, file_seed_list, save_path):  # 20221028 fred: changed args to file_path, file_seed_list, save_path
    file_handler = FileHandler()
    file_handler.set_path(save_path)  # 20221028 fred: changed  args.save_path to save_path

    # load some data to visualize from csv file
    df_seedlist = file_handler.read_file(
        file_seed_list)  # 20221028 fred: changed args.file_seed_list to file_seed_list
    df = file_handler.read_file(file_path)  # 20221028 fred: changed args.file_path to file_path

    env_name = (df_seedlist.iloc[:, 0:1].values[0])[0]
    train_demo_seeds = [int(x) for x in [df_seedlist.iloc[:, 1:].values[i] for i in range(0, df_seedlist.shape[0])]]

    # preprocess data
    sampler = Sampler()
    sample = sampler.sample_batchwise(df, len(train_demo_seeds), 100)

    final_loss = [round(float(x), 3) for x in [sample.iloc[:, :1].values[i] for i in range(0, sample.shape[0])]]
    final_pol_loss = [round(float(x), 3) for x in [sample.iloc[:, 1:2].values[i] for i in range(0, sample.shape[0])]]
    accuracy = [round(float(x), 3) for x in [sample.iloc[:, 2:3].values[i] for i in range(0, sample.shape[0])]]
    grad_norm = [round(float(x), 3) for x in [sample.iloc[:, 3:4].values[i] for i in range(0, sample.shape[0])]]
    zero = [int(x) for x in [sample.iloc[:, 4:5].values[i] for i in range(0, sample.shape[0])]]
    non_zero = [int(x) for x in [sample.iloc[:, 5:6].values[i] for i in range(0, sample.shape[0])]]

    predictions = sample.iloc[:, 8:9].values[0:]  # 20221102 fred: added to pass it to loss_reward.py

    a1 = sample.iloc[:, 6:7].values[0:]
    actions_pred = []
    for i in range(0, sample.shape[0]):
        a2 = a1[i, 0].lstrip('[').rstrip(']').split(',')
        a3 = [int(x) for x in a2 if x.strip().isdigit()]
        actions_pred.append(a3)

    demo_actions = sample.iloc[:, 7:8].values[0:]
    actions_true = []
    for i in range(0, sample.shape[0]):
        list1 = demo_actions[i, 0].lstrip('[').rstrip(']').split(',')
        list2 = [int(x) for x in list1 if x.strip().isdigit()]
        actions_true.append(list2)

    seed_list_enlarged = Calculator().calculate_missing_seeds(train_demo_seeds, len(actions_true))

    x = np.arange(len(grad_norm))
    (episodes_fail, episodes_success, all_episodes) = Filter().filter_episodes(env_name, x, actions_pred,
                                                                               seed_list_enlarged)

    global vis
    vis = PlotGrid()
    vis.set_env(env_name)
    plt.close()

    plotGrid = GradNormVis()
    UtilityVis().set_window_title(plotGrid.fig, "Feedback Processing")
    UtilityVis().set_title(plotGrid.fig, env_name, len(train_demo_seeds))
    plotGrid.set_plotdata(env_name, grad_norm, zero, non_zero, final_loss, actions_pred, actions_true,
                          seed_list_enlarged, accuracy, all_episodes, episodes_fail, episodes_success)

    global glyph, trajectory_glyph_path
    # pie_glyph_path
    glyph = Glyph()
    trajectory_glyph_path = glyph.make_glyph_folder("traj")
    # pie_glyph_path = glyph.make_glyph_folder("pie")
    glyph.set_glyph_max(len(grad_norm))
    glyph.init_display_list()

    # 20221109 fred: commented this out
    # plotGrid._grid(file_handler)

    # 20221028 fred: added this return:
    return (env_name, grad_norm, zero, non_zero, final_loss, actions_pred, predictions, actions_true,
            seed_list_enlarged, accuracy, all_episodes, episodes_fail, episodes_success, glyph, trajectory_glyph_path)

    # 20221028 fred: commented this out
    # if UtilityVis().type_of_script() != 'jupyter':
    #    plt.show(block=True)

    # 20221028 fred: commented this out

# if __name__ == "__main__":
#    args = parser.parse_args()
#    main(args)
