import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
import click


class PlotPaths(app.Canvas):
    def __init__(self, positions):
        self.VERT_SHADER = """
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        attribute vec3 a_position;
        void main (void) {
            gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
        }
        """

        self.FRAG_SHADER = """
        void main()
        {
            gl_FragColor = vec4(0,0,0,1);
        }
        """

        app.Canvas.__init__(self, keys='interactive', size=(400, 400))

        self.program_list = []
        self.translate = 5
        self.view = translate((0, 0, -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)

        for i in range(50):
            self.program_list.append(gloo.Program(self.VERT_SHADER, self.FRAG_SHADER))
            self.program_list[-1]['a_position'] = gloo.VertexBuffer(positions[:, i*3+1:i*3+3+1])
            self.program_list[-1]['u_projection'] = self.projection
            self.program_list[-1]['u_model'] = self.model
            self.program_list[-1]['u_view'] = self.view

        self.theta = 0
        self.phi = 0

        self.context.set_clear_color('white')
        self.context.set_state('translucent')

        self.timer = app.Timer('auto', connect=self.on_timer)

        self.show()

    # ---------------------------------
    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    # ---------------------------------
    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        for program in self.program_list:
            program['u_model'] = self.model
        self.update()

    # ---------------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 1.0, 1000.0)
        for program in self.program_list:
            program['u_projection'] = self.projection

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))
        for program in self.program_list:
            program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        for program in self.program_list:
            program.draw('line_strip')

    def on_mouse_move(self, event):
        print(event.delta)


    def on_mouse_press(self, event):
        print(event.pos)


if __name__ == '__main__':
    @click.command()
    @click.option('--param_file',prompt="File with parameters", help="File with parameters.")
    @click.option('--update_vector_file',prompt="File with update vector", help="File with update vector.")
    @click.option('--steps',prompt="Steps", help="Numper of steps")
    def main(param_file, update_vector_file, steps):
        shape = np.genfromtxt(param_file).astype(np.int)
        oxygen_path = np.memmap( update_vector_file, dtype='float32', mode='r+', shape=(shape[0], shape[1]*3-2))
        oxygen_path = oxygen_path[:int(steps)]

        c = PlotPaths(oxygen_path)
        app.run()

    main()

