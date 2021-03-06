from vispy import gloo
from vispy import app
from vispy.gloo import gl
import numpy as np
import os.path as path


class PlotItrium:
    def __init__(self, plot_paths):
        self.plot_paths = plot_paths

        self.filename = path.join(plot_paths.data_path, 'positions.xyz')

        self.positions = []
        with open(self.filename) as file_in:
            for line in file_in.readlines()[2:]:
                line = line.split('\t')
                if line[0] == 'Y':
                    self.positions.append((float(line[1]),
                                           float(line[2]),
                                           float(line[3])))

        self.positions = (np.array(self.positions) - plot_paths.delta_pos)+(.5, .5, .5)
        self.red = np.array((1, 0, 0, .75)).astype(np.float32).reshape(1, 4)

        VERT_SHADER = """
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;

        attribute vec3  a_position;
        attribute vec4  a_color;
        attribute float a_size;

        varying vec4 v_fg_color;
        varying vec4 v_bg_color;
        varying float v_radius;
        varying float v_linewidth;
        varying float v_antialias;

        void main (void) {
            v_radius = a_size;
            v_linewidth = 1.0;
            v_antialias = 1.0;
            v_fg_color  = vec4(0.0,0.0,0.0,0.5);
            v_bg_color  = a_color;

            gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
            gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
        }
        """

        FRAG_SHADER = """
        varying vec4 v_fg_color;
        varying vec4 v_bg_color;
        varying float v_radius;
        varying float v_linewidth;
        varying float v_antialias;
        void main()
        {
            float size = 2*(v_radius + v_linewidth + 1.5*v_antialias);
            float t = v_linewidth/2.0-v_antialias;
            float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
            float d = abs(r - v_radius) - t;
            if( d < 0.0 )
                gl_FragColor = v_fg_color;
            else
            {
                float alpha = d/v_antialias;
                alpha = exp(-alpha*alpha);
                if (r > v_radius)
                    gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
                else
                    gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
            }
        }
        """

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['u_model'] = self.plot_paths.model
        self.program['u_view'] = self.plot_paths.view
        self.program['u_projection'] = self.plot_paths.projection

        a_color, a_position, a_size = self.generate_points()

        self.program['a_color'] = gloo.VertexBuffer(a_color)
        self.program['a_position'] = gloo.VertexBuffer(a_position)
        self.program['a_size'] = gloo.VertexBuffer(a_size)

    def generate_points(self):
        a_color = [self.red for i in range(self.positions.shape[0])]
        a_position = self.positions
        a_size = [10 for i in range(self.positions.shape[0])]

        a_color = np.array(a_color).astype(np.float32)
        a_position = np.array(a_position).astype(np.float32)
        a_size = np.array(a_size).astype(np.float32)

        return a_color, a_position, a_size

    def on_draw(self, event):
        self.program.draw(gl.GL_POINTS)
