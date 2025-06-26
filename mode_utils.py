import os
import time


def set_end_str(path: str, new_end: str) -> str:
    end_str = f".{path.split('.')[-1]}"
    new_path = path[: -len(end_str)] + new_end
    return new_path


def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


class WDTagger:
    from tools.wd_tagger import WDTagger
    import threading

    def __init__(self, model_name: str = "wd-eva02-large-tagger-v3", dir: tuple[str, str] = ("./in", "./out"), confidence_threshold: float = 0.3, thread_count: int = 1):
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name

        self.tagger = self.WDTagger(
            self.model_name,
        )
        self.input_dir, self.output_dir = dir
        create_dir(self.output_dir)

        self.thread_count = thread_count

        self.thread_paths = self.__get_thrend_paths()

    def __get_paths(self) -> list[tuple[str, str]]:
        image_list = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    image_path = f"{root}/{file}"
                    out_path = image_path.replace(self.input_dir, self.output_dir)
                    out_path = set_end_str(out_path, ".txt")
                    image_list.append((image_path, out_path))
        return image_list

    def __get_thrend_paths(self) -> list[list[tuple[str, str]]]:
        image_list = self.__get_paths()
        thread_paths = []
        one_list_count = len(image_list) // self.thread_count
        ex_list_count = len(image_list) % self.thread_count
        for i in range(self.thread_count):
            if i < ex_list_count:
                thread_paths.append(image_list[i * (one_list_count + 1) : (i + 1) * (one_list_count + 1)])
            else:
                thread_paths.append(image_list[i * one_list_count : (i + 1) * one_list_count])
        return thread_paths

    def __run_one(self, task_id, start_time: float, paths: list[tuple[str, str]]) -> None:
        print(f"线程 {task_id} 开始执行")
        for image_path, out_path in paths:
            str_list = self.tagger.predict_confidence(image_path, self.confidence_threshold)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(", ".join(str_list))
            print(f"{image_path} -> {out_path}")
        end_time = time.time()
        print(f"线程 {task_id} 执行完毕, 耗时 {end_time - start_time:.2f} 秒")

    def run(self) -> None:
        start_time = time.time()
        threads = []
        for task_id in range(self.thread_count):
            thread = self.threading.Thread(target=self.__run_one, args=(task_id, start_time, self.thread_paths[task_id]))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        end_time = time.time()
        print(f"执行完毕, 共耗时 {end_time - start_time:.2f} 秒")


class ImageProcessing:
    from tools.image_processing import Image_Processing

    def __init__(self, input_dir: str, output_dir: str, image_long_size: int = 1024, is_resize: bool = True) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        create_dir(self.output_dir)
        self.image_long_size = image_long_size
        self.is_resize = is_resize
        self.image_paths = self.__get_paths()
        self.image_processing = self.Image_Processing()

    def __get_paths(self) -> list[tuple[str, str]]:
        image_list = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    image_path = f"{root}/{file}"
                    out_path = image_path.replace(self.input_dir, self.output_dir)
                    out_path = set_end_str(out_path, "[rario_name].png")
                    image_list.append((image_path, out_path))
        return image_list

    def run(self) -> None:
        for image_path, out_path in self.image_paths:
            new_image, ratio_name = self.image_processing.pro_image(image_path, self.image_long_size, self.is_resize)
            if new_image is None:
                continue
            if not self.is_resize:
                ratio_name = ""
            new_out_path = out_path.replace("[rario_name]", f"_{ratio_name}")
            new_image.save(new_out_path)
            print(f"{image_path} -> {new_out_path}")


class Txt2JsonProcessing:
    from tools.tagger_processing import txt2json
    import json

    def __init__(self, input_dir: str, output_dir: str) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        create_dir(self.output_dir)

        self.tagger_processing = self.txt2json()
        self.tag_paths = self.__get_paths()

    def __get_paths(self):
        txt_list = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".txt"):
                    txt_path = f"{root}/{file}"
                    out_path = txt_path.replace(self.input_dir, self.output_dir)
                    out_path = set_end_str(out_path, ".json")
                    txt_list.append((txt_path, out_path))
        return txt_list

    def run(self):
        for txt_path, out_path in self.tag_paths:
            data = self.tagger_processing.pro_tagger(txt_path)
            with open(out_path, "w", encoding="utf-8") as f:
                self.json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"{txt_path} -> {out_path}")


class Json2TxtProcessing:
    from tools.tagger_processing import json2txt

    def __init__(self, input_dir: str, output_dir: str, processsing_py: str = "./default_pro_json.py") -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        create_dir(self.output_dir)

        self.tagger_json_processing = self.json2txt(processsing_py)
        self.tag_paths = self.__get_paths()

    def __get_paths(self):
        json_list = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".json"):
                    json_path = f"{root}/{file}"
                    out_path = json_path.replace(self.input_dir, self.output_dir)
                    out_path = set_end_str(out_path, ".txt")
                    json_list.append((json_path, out_path))
        return json_list

    def run(self):
        for json_path, out_path in self.tag_paths:
            data = self.tagger_json_processing.pro_json(json_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(", ".join(data))
            print(f"{json_path} -> {out_path}")


class YoloCroppr:
    from tools.yolo import ImageCropper
    from PIL import Image

    def __init__(self, input_dir: str, output_dir: str, cropers_names=["person", "halfbody", "head", "face"], confidence_threshold: float = 0.35, iou_threshold: float = 0.7) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        create_dir(self.output_dir)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.cropers_names = cropers_names

        self.cropers = {}
        for name in self.cropers_names:
            self.cropers[name] = self.ImageCropper(name, self.confidence_threshold, self.iou_threshold)

        self.old_image_paths = self.__get_old_paths()
        self.yolo_data = self.__get_yolo_data()

    def get_croper(self, name: str) -> ImageCropper:
        return self.ImageCropper(name, self.confidence_threshold, self.iou_threshold)

    def __get_old_paths(self):
        paths = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    image_path = f"{root}/{file}"
                    out_path = image_path.replace(self.input_dir, self.output_dir)
                    out_path = set_end_str(out_path, "[out_name].png")
                    paths.append((image_path, out_path))
        return paths

    def __get_yolo_data(self):
        data = []
        for image_path, out_path in self.old_image_paths:
            images = [("origin", out_path.replace("[out_name]", f"_origin"), None)]
            for key, croper in self.cropers.items():
                start_time = time.time()
                results = croper.get_image_results(image_path)
                key_int = 1
                for result in results:
                    class_name = result["class_name"]
                    if key_int > 1:
                        new_name = f"{class_name}_{key_int}"
                    else:
                        new_name = class_name
                    images.append((key, out_path.replace("[out_name]", f"_{new_name}"), result))
                    key_int = key_int + 1
                end_time = time.time()
                print(f"{image_path} 的 {key} 处理,耗时 {end_time - start_time:.2f} 秒")
            data.append((image_path, images))
        return data

    def __check_box_ratio(self, box, image_pil):
        image_width, image_height = image_pil.size
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        image_area = image_width * image_height
        ratio = box_area / image_area
        return ratio

    def __crop_and_save(self, images: tuple[str, str, list[dict]]):
        image_path, images = images
        for key, out_path, result in images:
            image_pil = self.Image.open(image_path)
            if result is not None:
                box = result["box"]
                if self.__check_box_ratio(box, image_pil) > 0.7:
                    continue
                new_image = image_pil.crop(box)
            else:
                new_image = image_pil
            new_image.save(out_path)
            print(f"{image_path} -> {out_path}")

    def run(self):
        for data in self.yolo_data:
            self.__crop_and_save(data)
