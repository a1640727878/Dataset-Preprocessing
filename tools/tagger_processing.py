import os
import re
import json
import importlib.util
from pathlib import Path


def dict2flat(dict_data: dict) -> dict:
    new_data = {}

    def __toflat(data: dict, parent_key=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                __toflat(value, new_key)
        elif isinstance(data, list):
            new_data[parent_key] = data

    __toflat(dict_data)
    return new_data


def flat2dict(flat_data: dict) -> dict:
    new_data = {}

    for key, value in flat_data.items():
        keys = key.split(".")
        new_data_2 = new_data

        for key in keys[:-1]:
            new_data_2 = new_data_2.setdefault(key, {})
        key = keys[-1]
        new_data_2[key] = value

    return new_data


class txt2json:

    def __init__(self, while_count=5) -> None:
        self.data_json_dir = "./data/tagger_json"

        self.general_json_path = f"{self.data_json_dir}/general.json"
        self.general_data = self.__json2dict(self.general_json_path)

        self.tools_json_dir = f"{self.data_json_dir}/tools"
        self.tools_json = {}

        self.output_data = {}

        for json_path in Path(self.tools_json_dir).rglob("*.json"):
            json_path_str = str(json_path)
            json_data = self.__json2dict(json_path_str)
            if isinstance(json_data, dict):
                json_data = dict2flat(json_data)
            json_name = json_path_str[self.tools_json_dir.__len__() - 1 : -5].replace("\\", "/")
            self.tools_json[json_name] = json_data
        print(self.tools_json)

        self.while_count = while_count

    def __json2dict(self, json_path: str) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        return json_data

    def __txt2dict(self, txt_path: str) -> list[str]:
        result = []
        with open(txt_path, "r", encoding="utf-8") as file_buffer:
            for line in file_buffer:
                strs = line.strip().replace("\n", "").replace(", ", ",").split(",")
                result.extend(strs)
        return result

    def __get_tools_list(self, name: str):
        path, _, key = name.partition(".")
        tools_dict = self.tools_json.copy().get(path, [])
        if isinstance(tools_dict, list):
            return tools_dict
        return tools_dict[key]

    def __get_general_data_copy(self) -> dict:
        return self.general_data.copy()

    def __remove_txtlist(self, data_dict: dict, txt_list: list[str], str_start: tuple[str, ...] = ("$", "#")) -> tuple[list[str], dict]:
        new_txt_list = txt_list.copy()
        flat_data = dict2flat(data_dict)
        for key, value in flat_data.items():
            new_data = []
            for item in value:
                item = str(item)
                if item.startswith(str_start) or item in new_txt_list:
                    new_data.append(item)
                if item in new_txt_list:
                    new_txt_list.remove(item)
            flat_data[key] = new_data
        return new_txt_list, flat2dict(flat_data)

    def __updata_placeholder(self, data: dict, txt_list: list[str] = []):
        flat = dict2flat(data)
        for key, value in flat.items():
            new_data = []
            for item in value:
                item = str(item)
                if not item.startswith("#"):
                    new_data.append(item)
                    continue
                item_str = item[1:]
                list_tools_names = re.findall(r"\{(.*?)\}", item_str)
                item_list = [item_str]
                for name in list_tools_names:
                    new_item_list = []
                    tools_list = self.__get_tools_list(name)
                    for tool_str in tools_list:
                        for tool in item_list:
                            new_item = str(tool).replace(r"{" + name + r"}", tool_str)
                            new_item_list.append(new_item)
                    item_list = new_item_list.copy()
                new_data.extend(item_list)
            new_data_2 = []
            for item in new_data:
                item = str(item)
                if item.startswith(("$", "#")) or item in txt_list:
                    new_data_2.append(item)
            flat[key] = new_data_2
        return flat2dict(flat)

    def __updata_replace(self, data: dict, txt_list: list[str] = [], remove_txt_list: list[str] = []):
        flat = dict2flat(data)
        for key, value in flat.items():
            new_data = []
            for item in value:
                item = str(item)
                if not item.startswith("$"):
                    new_data.append(item)
                    continue
                for tag in remove_txt_list:
                    if re.match(item[1:], tag):
                        new_data.append(tag)
            new_data_2 = []
            for item in new_data:
                item = str(item)
                if item.startswith(("$", "#")) or item in txt_list:
                    new_data_2.append(item)
            flat[key] = new_data_2
        return flat2dict(flat)

    def pro_tagger(self, txt_path: str) -> dict:
        general_data = self.__get_general_data_copy()
        txt_list = self.__txt2dict(txt_path)

        while_count = self.while_count
        while while_count > 0:
            while_count -= 1
            new_general_data = self.__updata_placeholder(general_data, txt_list)
            general_data = new_general_data

        remove_txt_list, general_data_2 = self.__remove_txtlist(general_data, txt_list, ("$"))
        general_data = general_data_2.copy()

        while_count = self.while_count
        while while_count > 0:
            while_count -= 1
            new_general_data = self.__updata_replace(general_data, txt_list, remove_txt_list)
            general_data = new_general_data

        remove_txt_list, general_data_2 = self.__remove_txtlist(general_data, txt_list, ())
        general_data = general_data_2.copy()

        general_data["misc"] = remove_txt_list
        return general_data


class json2txt:
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
        flat_data = dict2flat(data)
        for key, value in flat_data.items():
            for item in value:
                new_txt_list.append(item)
        return new_txt_list


class Default_Json_Processing:

    def get_data(self, json_data: dict):
        return json_data
