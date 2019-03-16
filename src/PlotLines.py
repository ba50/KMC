import numpy as np
from vispy import gloo


class PlotLines:
    def __init__(self, plot_paths):
        self.plot_paths = plot_paths
        self.step = 5000

        VERT_SHADER = """
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
        FRAG_SHADER = """
        varying vec4 v_color;
        void main()
        {
            gl_FragColor = v_color;
        }
        """

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['u_model'] = self.plot_paths.model
        self.program['u_view'] = self.plot_paths.view

        a_color, a_position = self.generate_paths()

        self.program['a_color'] = gloo.VertexBuffer(a_color)
        self.program['a_position'] = gloo.VertexBuffer(a_position)

    # ---------------------------------
    def generate_paths(self):
        a_position = []
        a_color = []
        for i in range(self.plot_paths.atoms_number):
            tmp = self.make_line(i)
            a_color.append(tmp[0])
            a_position.append(tmp[1])

        a_color = np.array(a_color)
        a_position = np.array(a_position)
        return a_color, a_position

    # ---------------------------------
    def make_line(self, index):
        line = self.plot_paths.positions[:self.plot_paths.animation_step, index]
        line = np.array(line).astype(np.float32)

        color = [self.plot_paths.alpha]
        for j in range(1, self.plot_paths.animation_step-1):
            color.append(self.plot_paths.a_color[index])
        color.append(self.plot_paths.alpha)
        color = np.array(color).astype(np.float32)
        return color, line

    # ---------------------------------
    def on_key_press(self, event):
        if event.key == 'Up':
            self.step = np.clip(self.step+100,
                                2,
                                self.plot_paths.positions.shape[0])
            print('Delta: ~', self.step)
        if event.key == 'Down':
            self.step = np.clip(self.step-100, 
                                2,
                                self.plot_paths.positions.shape[0])
            print('Delta steps: ', self.step)

        if event.key == 'Left':
            self.plot_paths.animation_step = np.clip(self.plot_paths.animation_step-self.step,
                                                     2,
                                                     self.plot_paths.positions.shape[0])
        if event.key == 'Right':
            self.plot_paths.animation_step = np.clip(self.plot_paths.animation_step+self.step,
                                                     2, 
                                                     self.plot_paths.positions.shape[0]) 

        a_color, a_position = self.generate_paths()

        self.program['a_color'] = gloo.VertexBuffer(a_color)
        self.program['a_position'] = gloo.VertexBuffer(a_position)
        print("Time: {:.2f} [ps]".format(self.plot_paths.time[self.plot_paths.animation_step]))

    # ---------------------------------
    def on_draw(self, event):
            self.program.draw('line_strip')
