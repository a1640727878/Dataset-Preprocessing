import re


def flat_dict_encoder(dict_data: dict) -> dict:
    data_2 = {}

    def __flatten(data, parent_key=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                __flatten(value, new_key)
        elif isinstance(data, list):
            data_2[parent_key] = data

    __flatten(dict_data)
    return data_2


def flat_dict_decoder(dict_data: dict) -> dict:
    data_2 = {}

    for flat_key, value in dict_data.items():
        keys = flat_key.split(".")
        new_data = data_2

        for key in keys[:-1]:
            new_data = new_data.setdefault(key, {})
        flat_key = keys[-1]
        new_data[flat_key] = value

    return data_2


def get_data(json_data: dict):
    flat_dict = flat_dict_encoder(json_data.copy())
    new_data = {}

    for key, value in flat_dict.items():
        new_value = []
        if value == []:
            continue
        elif key == "blacklist":
            continue
        elif key == "rating" and len(value) > 1:
            new_value.append(value[0])
        elif key == "character" and "solo focus" in value:
            for item in value:
                result = re.search(r"^[2-6](.*)", item)
                if not result:
                    new_value.append(item)
        else:
            for item in value:
                new_value.append(item)

        new_data[key] = new_value

    return flat_dict_decoder(new_data)
