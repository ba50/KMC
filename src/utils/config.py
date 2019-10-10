from pathlib import Path


def get_config(path: Path):
    str_index = [0]
    float_index = [*range(1, 16)]
    with path.open() as _config:
        lines = _config.readlines()
    _config = {}
    for index, line in enumerate(lines):
        if index in str_index:
            _data = line.split('#')
            _key = _data[1].strip()
            _config[_key] = _data[0].strip()

        if index in float_index:
            _data = line.split('#')
            _key = _data[1].strip()
            _config[_key] = float(_data[0].strip())

    return _config
