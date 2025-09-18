# Dataset-Preprocessing

## 功能
 - [x] GUI 界面  
 - [x] 规整图片比例  
 - [x] wd_tagger 打标  
 - [x] yolo 裁剪图片  
 - [ ] 整理 wd_tagger 标签 (旧版写得太差, 不适配)  

## 使用说明

自形安装 python3.10 - 3.12, 使用 'pip intall -r requirements.txt' 安装依赖

```bash
python main.py
```

使用以上命令运行小工具


## 感谢

- [Deepghs](https://huggingface.co/deepghs) 感谢 Deepghs 提供的大部分模型 的 onnx 格式  
- [SmilingWolf](https://huggingface.co/SmilingWolf) *wd_tagger 模型的原作者  
- [Camais03](https://huggingface.co/Camais03) *camie 模型的原作者  
- [Pixai](https://huggingface.co/pixai-labs/pixai-tagger-v0.9) *pixai 提供的最新打标模型  
- [Nagadomi](https://github.com/nagadomi/nunif) *超分模型的原作者  
- 最后感谢来源神秘的 Z3D-E621-Convnext 模型 和 他的搬运 [models_url](https://huggingface.co/toynya/Z3D-E621-Convnext/)  