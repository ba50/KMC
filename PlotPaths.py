import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
import click

from multiprocessing import Pool


class PlotPaths(app.Canvas):
    def __init__(self, positions, atom_number=50, resolution=(800,600)):
        self.positions = positions #- positions[:1].mean()
        self.atom_number = atom_number
        self.resolution = resolution

        self.mouse_press_point = 0, 0
        self.mouse_press = False
        self.theta = 0
        self.phi = 0
        self.delta_theta = 0
        self.delta_phi = 0
        self.animation_step = 1

        self.VERT_SHADER = """
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        attribute vec3 a_position;
        attribute vec4 a_color;
        varying vec4 v_color;
        void main (void) {
            v_color = a_color;
            gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
        }
        """

        self.FRAG_SHADER = """
        varying vec4 v_color;
        void main()
        {
            gl_FragColor = v_color;
        }
        """

        app.Canvas.__init__(self, keys='interactive', size=self.resolution)

        self.translate = 50
        self.view = translate((0, 0, -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)

        self.program = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        a_positons = np.copy(self.positions[:self.animation_step, 1:])
        print(a_positons.shape)
        exit()
        self.program['a_position'] = gloo.VertexBuffer(a_positons)
        self.a_color = np.append(np.random.rand(3), 1).reshape(1,4)
        self.program['a_color'] = gloo.VertexBuffer([self.a_color[-1] for i in range(self.animation_step)])
        self.program['u_projection'] = self.projection
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        self.context.set_clear_color('white')
        self.context.set_state('translucent')

        self.timer = app.Timer('auto', connect=self.on_timer)
        self.timer.start()

        self.show()

    # ---------------------------------
    def on_key_press(self, event):
        step = 1000
        if event.key == 'Left':
            print(self.animation_step)
            self.animation_step = np.clip(self.animation_step-step, 1, self.positions.shape[0])

        if event.key == 'Right':
            print(self.animation_step)
            self.animation_step = np.clip(self.animation_step+step, 1, self.positions.shape[0])

        self.program['a_position'] = gloo.VertexBuffer(np.copy(self.positions[:self.animation_step, 1:atom_number*3+3+1]))
        self.program['a_color'] = gloo.VertexBuffer([self.a_color for i in range(self.animation_step)])

    # ---------------------------------
    def on_timer(self, event):
        self.phi += np.clip(self.delta_phi, -1, 1)
        self.theta += np.clip(self.delta_theta, -1, 1)
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        self.update()

    # ---------------------------------
    def on_resize(self, event):
        self.resolution = event.physical_size
        gloo.set_viewport(0, 0, self.resolution[0], self.resolution[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 1.0, 1000.0)
        for program in self.program_list:
            program['u_projection'] = self.projection

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        self.program.draw('line_strip')

    # ---------------------------------
    def on_mouse_move(self, event):
        print(self.program_list[0]['u_model'], self.model)
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
    @click.option('--steps',prompt="Steps", help="Numper of steps")
    def main(param_file, update_vector_file, steps):
        shape = np.genfromtxt(param_file).astype(np.int)
        oxygen_path = np.load(update_vector_file)
        print(oxygen_path.shape)
        exit()
        oxygen_path = oxygen_path[:int(steps)]
        c = PlotPaths(oxygen_path, 10)
        app.run()

    main()

