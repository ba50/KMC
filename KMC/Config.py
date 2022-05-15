from pathlib import Path


class Config:
    def __init__(self, inputs) -> None:
        self.cell_type = inputs["cell_type"]
        self.size = {
            "x": inputs["size_x"],
            "y": inputs["size_y"],
            "z": inputs["size_z"],
        }
        self.thermal = inputs["thermal"]
        self.time_start = inputs["time_start"]
        self.time_end = inputs["time_end"]
        self.window = inputs["window"]
        self.window_epsilon = inputs["window_epsilon"]
        self.contact_switch_left = inputs["contact_switch_left"]
        self.contact_switch_right = inputs["contact_switch_right"]
        self.contact_left = inputs["contact_left"]
        self.contact_right = inputs["contact_right"]
        self.amplitude = inputs["amplitude"]
        self.frequency = inputs["frequency"]
        self.periods = inputs["periods"]
        self.static_potential = inputs["static_potential"]
        self.temperature_scale = inputs["temperature_scale"]

    def save(self, save_path: Path):
        with (save_path / "input.kmc").open("w") as file_out:
            file_out.write("{}\t# cell_type\n".format(self.cell_type.lower()))
            file_out.write("{}\t# size_x\n".format(self.size["x"]))
            file_out.write("{}\t# size_y\n".format(self.size["y"]))
            file_out.write("{}\t# size_z\n".format(self.size["z"]))
            file_out.write(
                "{}\t# thermalization_time\n".format(self.thermal)
            )
            file_out.write("{}\t# time_start\n".format(self.time_start))
            file_out.write("{}\t# time_end\n".format(self.time_end))
            file_out.write("{}\t# window\n".format(self.window))
            file_out.write("{}\t# window_epsilon\n".format(self.window_epsilon))
            file_out.write(
                "{}\t# contact_switch_left\n".format(self.contact_switch_left)
            )
            file_out.write(
                "{}\t# contact_switch_right\n".format(self.contact_switch_right)
            )
            file_out.write("{}\t# contact_left\n".format(self.contact_left))
            file_out.write("{}\t# contact_right\n".format(self.contact_right))
            file_out.write("{}\t# amplitude\n".format(self.amplitude))
            file_out.write("{}\t# frequency\n".format(self.frequency))
            file_out.write("{}\t# periods\n".format(self.periods))
            file_out.write("{}\t# static_potential\n".format(self.static_potential))
            file_out.write("{}\t# temperature_scale\n".format(self.temperature_scale))

    @staticmethod
    def load(load_path: Path):
        str_index = [0]
        float_index = [*range(1, 9)]
        bool_index = [9, 10]
        float_index.extend([*range(11, 18)])
        with load_path.open() as config:
            lines = config.readlines()
        config = {}
        for index, line in enumerate(lines):
            if index in str_index:
                data = line.split("#")
                key = data[1].strip()
                config[key] = data[0].strip()

            if index in bool_index:
                data = line.split("#")
                key = data[1].strip()
                config[key] = bool(data[0].strip())

            if index in float_index:
                data = line.split("#")
                key = data[1].strip()
                config[key] = float(data[0].strip())

        return Config(config)
