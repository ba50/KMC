from pathlib import Path

from src.GenerateXYZ import GenerateXYZ
from src.utils.config import get_config


if __name__ == "__main__":
    sim_path = Path("F:\\KMC_data\\data_2021_03_23_v0\\11_7_7_random_0_a_0_1.0")
    data_path = sim_path / 'oxygen_map\\positions_copy.xyz'

    config = get_config(sim_path / 'input.kmc')

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(data_path)
    delete_index = raw_frames['frame'].unique()

    filter_index = [0]
    last_frame = raw_frames[raw_frames['frame'] == 0]
    for index in delete_index[1:]:
        next_frame = raw_frames[raw_frames['frame'] == index]
        last_pos = last_frame[['x', 'y', 'z']].values
        next_pos = next_frame[['x', 'y', 'z']].values
        delta = abs((next_pos - last_pos).sum())
        if delta > 1:
            print(delta)
            filter_index.append(index)
        last_frame = next_frame

    raw_frames = raw_frames[raw_frames['frame'].apply(lambda x: x in filter_index)]
    GenerateXYZ.write_frames_from_dataframe(data_path.parent/'positions_filter.xyz', raw_frames, num_atoms)
