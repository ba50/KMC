import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
import click

from PlotLines import PlotLines
from PlotPoints import PlotPoints


class PlotPaths(app.Canvas):
    def __init__(self, positions, atoms_number=50, resolution=(800,600)):
        self.resolution = resolution
        app.Canvas.__init__(self, keys='interactive', size=self.resolution)
        self.positions = positions - positions[:1].mean()
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
        self.programs = [self.plot_lines.program, self.plot_points.program]

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
    @click.option('--param_file',prompt="File with parameters", help="File with parameters.")
    @click.option('--update_vector_file',prompt="File with update vector", help="File with update vector.")
    def main(param_file, update_vector_file):
        shape = np.genfromtxt(param_file).astype(np.int)
        oxygen_path = np.load(update_vector_file)
        oxygen_path = oxygen_path.reshape(shape[0], shape[1], 3)
        c = PlotPaths(oxygen_path, shape[1])
        app.run()

    main()

