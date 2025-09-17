import yaml
from pathlib import Path
import copy


class Config(dict):
    """
    This Config class extends a normal dictionary with the getattr and setattr functionality, and it enables saving to a yml.
    """
    def __init__(self, dictionary):
        super().__init__(dictionary)

        for key, value in self.items():
            assert(key not in ['keys', 'values', 'items']), 'The configuration contains the following key {key} which is reserved already as a standard attribute of a dict.'
            
            if isinstance(value, list) or isinstance(value, tuple):
                detected_dict = 0
                for idx, val in enumerate(value):
                    if isinstance(val, dict):
                        val = Config(val)
                        self[key][idx] = val
                        #setattr(self, key, val)
                        detected_dict += 1
                if not detected_dict:
                    setattr(self, key, value)

            elif isinstance(value, dict):
                value = Config(value)
                setattr(self, key, value)
            else:
                setattr(self, key, value)
            

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getstate__(self):
        return self

    def serialize(self):
        dictionary = {}
        for key, value in self.items():
            if isinstance(value, Config):
                dictionary[key] = value.serialize()
            else:
                dictionary[key] = value
        return dictionary

    def deep_copy(self):
        return Config(copy.deepcopy(self.serialize()))

    def save_to_yaml(self, path):
        with open(Path(path), 'w') as save_file:
            yaml.dump(self.serialize(), save_file, default_flow_style=False)

def load_config_from_yaml(path):
    dictionary = yaml.load(open(Path(path)), Loader=yaml.FullLoader)
    if dictionary:
        return Config(dictionary)
    else:
        return {}
