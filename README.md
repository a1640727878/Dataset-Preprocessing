# Dataset-Preprocessing

## 功能计划
- [x] yolo 截取角色图片
- [x] 规整化图片
- [x] wd_tagger 对图片生成 txt 格式打标
- [x] txt 格式打标结果整理成 json 格式
- [x] 对json格式标签批处理转回txt格式
- [ ] GUI 界面
  
## 使用教程

### **yolo批量裁剪图**

```bash
python main.py --mode yolo_crop --input_dir [输入文件夹] --output_dir [输出文件夹]
```

### **规整化图片**

```bash
python main.py --mode pro_image --input_dir [输入文件夹] --output_dir [输出文件夹] --size [输出图片大小]
```

### **打标**

```bash
python main.py --mode wd-tagged --wd_model [模型] --input_dir [输入文件夹] --output_dir [输出文件夹] --confidence_threshold [置信度阈值] --thread_count [线程数]
```

### **txt标签批处理**

```bash
python main.py --mode pro_tagger --input_dir [输入文件夹] --output_dir [输出文件夹]
```

### **json标签逆向批处理**

```bash
python main.py --mode pro_json --input_dir [输入文件夹] --output_dir [输出文件夹] --processsing_py [处理脚本]
```

## 感谢
- [Deepghs](https://huggingface.co/deepghs)  *yolo模型的来源,
- [SmilingWolf](https://huggingface.co/SmilingWolf) *wd-tagger的来源
- [Nagadomi](https://github.com/nagadomi/nunif) *超分模型的来源
