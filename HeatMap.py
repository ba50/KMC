import sys

import click
import numpy as np

from vispy import scene
from vispy.visuals.transforms import STTransform

class HeatMap:
    def __init__(self, data_in):
        self.array = []
        with open(data_in) as heat_map_file:
            for line in heat_map_file:
                self.array.append([int(word) for word in line.split()])
        self.canvas = None

    def plot(self):

        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                                   size=(800, 600), show=True)

        view = self.canvas.central_widget.add_view()
        view.camera = 'fly'

        positions = []
        for pos in self.array:
            if pos[3] != 0:
                positions.append(scene.visuals.Sphere(
                    radius=pos[3]/10,
                    method='latitude',
                    parent=view.scene,
                    edge_color='black'))
                positions[-1].transform = STTransform(translate=[pos[0], pos[1], pos[2]])

        view.camera.set_range(x=[-3, 3])


if __name__ == '__main__' and sys.flags.interactive == 0:
    @click.command()
    @click.option('--data_in', prompt="Heat map data in: ", help="Heat map data in: ") 
    def main(data_in):
        heat_map = HeatMap(data_in)
        heat_map.plot()
        heat_map.canvas.app.run()

    main()

