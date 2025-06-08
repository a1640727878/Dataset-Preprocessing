import importlib.util
import json
import os
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


class Tagger_Processing:
    def __init__(self, data_path: str = "./data/wd_tagger_data", while_count: int = 5):
        self.data_path = data_path

        self.general_path = f"{self.data_path}/general.json"
        self.tools_path = f"{self.data_path}/tools"

        self.general = self.__parse_json(self.general_path)
        self.tools = {}

        for root, _, files in os.walk(self.tools_path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    replace_str = file_path.replace("\\", "/").replace(".json", "").replace(f"{self.tools_path}/", "")
                    self.tools[replace_str] = self.__parse_json(file_path)

        self.while_count = while_count

    def __parse_json(self, json_path: str) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __pares_txt(self, txt_path: str) -> list[str]:
        with open(txt_path, "r", encoding="utf-8") as f:
            str = f.read().replace(", ", ",")
        str_list = str.split(",")
        return str_list

    def __remove_txtlist(self, data_dict: dict, txt_list: list[str], str_start: list[str] = []) -> tuple[list[str], dict]:
        txt_list_2 = txt_list.copy()
        flat_dict = flat_dict_encoder(data_dict)
        for key, value in flat_dict.items():
            new_data = []
            for item in value:
                bools = []
                for str in str_start:
                    bools.append(item.startswith(str))
                result = any(bools)
                if result or item in txt_list_2:
                    new_data.append(item)
                if item in txt_list_2:
                    txt_list_2.remove(item)
            flat_dict[key] = new_data
        return txt_list_2, flat_dict_decoder(flat_dict)

    def __get_tools_data(self, dict_path: str):
        data = self.tools.copy()
        for path in dict_path.split("."):
            if data is None:
                continue
            data = data.get(path)
        return data

    def __set_placeholder(self, data: dict or list, txt_list: list[str] = []) -> dict or list:
        flat_dict = flat_dict_encoder(data)
        for key, value in flat_dict.items():
            new_data = []
            for item in value:
                if not item.startswith("#"):
                    new_data.append(item)
                    continue
                item_str = item[1:]
                list_tools_names = re.findall(r"\{(.*?)\}", item_str)
                list_tools = [item_str]
                for name in list_tools_names:
                    tools_data_list = self.__get_tools_data(name)
                    if tools_data_list is None:
                        continue
                    new_list_tools = []
                    for tool_str in tools_data_list:
                        for tool in list_tools:
                            new_tools_str = tool.replace(f"{{{name}}}", tool_str)
                            new_list_tools.append(new_tools_str)
                    list_tools = new_list_tools
                for tool in list_tools:
                    new_data.append(tool)
            new_data_2 = []
            for item in new_data:
                if item.startswith("$") or item.startswith("#") or item in txt_list:
                    new_data_2.append(item)
            flat_dict[key] = new_data_2
        return flat_dict_decoder(flat_dict)

    def __set_replace(self, data: dict or list, txt_list: list[str] = []) -> dict or list:
        flat_dict = flat_dict_encoder(data)
        for key, value in flat_dict.items():
            new_data = []
            for item in value:
                if not item.startswith("$"):
                    new_data.append(item)
                    continue
                item_str = item[1:]
                for tag in txt_list:
                    if re.match(item_str, tag):
                        new_data.append(tag)
            new_data_2 = []
            for item in new_data:
                if item.startswith("$") or item.startswith("#") or item in txt_list:
                    new_data_2.append(item)
            flat_dict[key] = new_data_2
        return flat_dict_decoder(flat_dict)

    def pro_tagger(self, txt_path: str) -> dict:
        general_data = self.general.copy()
        txt_list = self.__pares_txt(txt_path)

        while_count = self.while_count
        while while_count > 0:
            while_count -= 1
            new_general_data = self.__set_placeholder(general_data, txt_list.copy())
            general_data = new_general_data

        txt_list_2, general_data_2 = self.__remove_txtlist(general_data, txt_list, ["$"])
        general_data = general_data_2.copy()

        while_count = self.while_count
        while while_count > 0:
            while_count -= 1
            new_general_data = self.__set_replace(general_data, txt_list.copy())
            general_data = new_general_data

        txt_list_2, general_data_2 = self.__remove_txtlist(general_data, txt_list, ["&"])
        general_data = general_data_2.copy()

        general_data["misc"] = txt_list_2
        return general_data


class Tagger_Json_Processing:

    def __init__(self, processsing_py: str = "./default_pro_json.py"):
        self.processsing_py = processsing_py
        self.module = None
        if not os.path.isfile(self.processsing_py):
            self.module = Default_Json_Processing()
        else:
            module_name = os.path.splitext(os.path.basename(self.processsing_py))[0]
            spec = importlib.util.spec_from_file_location(module_name, self.processsing_py)
            if spec is None:
                print(f"Failed to load module from {self.processsing_py}")
                self.module = Default_Json_Processing()
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

            if not hasattr(self.module, "get_data") or not callable(self.module.get_data):
                print(f"Module {self.processsing_py} does not have a function named 'get_data'")
                self.module = Default_Json_Processing()

    def __get_data(self, json_data: dict):
        return self.module.get_data(json_data)

    def pro_json(self, json_path: str) -> list[str]:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        data = self.__get_data(json_data)
        new_txt_list = []
        flat_data = flat_dict_encoder(data)
        for key, value in flat_data.items():
            for item in value:
                new_txt_list.append(item)
        return new_txt_list


class Default_Json_Processing:

    def get_data(self, json_data: dict):
        return json_data


if __name__ == "__main__":
    tagger_processing = Tagger_Processing()
