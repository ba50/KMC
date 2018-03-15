import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
import click
import os.path as path

from PlotLines import PlotLines
from PlotPoints import PlotPoints
from PlotItrium import PlotItrium


class PlotPaths(app.Canvas):
    def __init__(self, time, positions, path_to_data, file_name, atoms_number=50, resolution=(800,600)):
        self.time = time
        self.path_to_data = path_to_data
        self.file_name = file_name
        self.resolution = resolution
        app.Canvas.__init__(self, keys='interactive', size=self.resolution)
        self.delta_pos = positions[:1].mean()
        self.positions = positions - self.delta_pos
        self.atoms_number = atoms_number

        self.mouse_press_point = 0, 0
        self.mouse_press = False
        self.theta = 0
        self.phi = 0
        self.delta_theta = 0
        self.delta_phi = 0
        self.animation_step = 2

        self.translate = 50
        self.view = translate((0, 0, -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)

        self.alpha = np.array((0, 0, 0, 0)).astype(np.float32).reshape(1, 4)
        self.a_color = [np.append(np.random.rand(3), 1).astype(np.float32).reshape(1,4) for i in range(self.positions.shape[1])]

        self.plot_lines = PlotLines(self)
        self.plot_points = PlotPoints(self)
        self.plot_itrium = PlotItrium(self)
        self.programs = [self.plot_lines.program,
                         self.plot_points.program,
                         self.plot_itrium.program]

        self.context.set_clear_color('white')
        self.context.set_state('translucent')

        self.timer = app.Timer('auto', connect=self.on_timer)
        self.timer.start()

        self.show()

    # ---------------------------------
    def on_key_press(self, event):
        self.plot_lines.on_key_press(event)

    # ---------------------------------
    def on_timer(self, event):
        self.phi += np.clip(self.delta_phi, -1, 1)
        self.theta += np.clip(self.delta_theta, -1, 1)
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        for program in self.programs:
            program['u_model'] = self.model

        self.update()

    # ---------------------------------
    def on_resize(self, event):
        self.resolution = event.physical_size
        gloo.set_viewport(0, 0, self.resolution[0], self.resolution[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 1.0, 1000.0)
        for program in self.programs:
            program['u_projection'] = self.projection

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))
        for program in self.programs:
            program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        self.plot_lines.on_draw(event)
        self.plot_points.on_draw(event)
        self.plot_itrium.on_draw(event)

    # ---------------------------------
    def on_mouse_move(self, event):
        if self.mouse_press:
            speed = .001
            self.delta_phi += (event.pos[0] - self.mouse_press_point[0])*speed
            self.delta_theta += (event.pos[1] - self.mouse_press_point[1])*speed

    # ---------------------------------
    def on_mouse_press(self, event):
        self.mouse_press = True
        self.mouse_press_point = event.pos

    # ---------------------------------
    def on_mouse_release(self, event):
        self.mouse_press = False
        self.delta_phi = 0
        self.delta_theta = 0


if __name__ == '__main__':
    @click.command()
    @click.option('--path_to_data',prompt="Path to data", help=" Path to data.")
    @click.option('--file_name',prompt="Filename", help="Filename.")
    def main(path_to_data, file_name):
        param_file = path.join(path_to_data, 'param_'+file_name+'.dat')
        time_file = path.join(path_to_data, 'time_vector_'+file_name+'.npy')
        data_file = path.join(path_to_data, 'update_vector_'+file_name+'.npy')

        shape = np.genfromtxt(param_file).astype(np.int)
        oxygen_path = np.load(data_file, mmap_mode='r')
        time = np.load(time_file)
        oxygen_path = oxygen_path.reshape(shape[0], shape[1], 3)
        c = PlotPaths(time, oxygen_path, path_to_data, file_name, shape[1])
        app.run()

    main()

