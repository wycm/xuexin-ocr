# xuexin-ocr
*  xuexin-ocr是一个针对学信网(https://www.chsi.com.cn/)学籍&学历图片信息内容识别项目
## 识别效果效果
![](https://github.com/wycm/xuexin-ocr/blob/master/1.png)

## 使用

### 模型训练

1. 样本字体图片生成：Run with gen_printed_chinese_char.py。字体相关参数使用默认参数，这组参数经过我多次调整后准确率相对较高的一组参数。样本集见label_dict.py,这里采用了6000+个汉字和数字等字符。其中字体文件在checkfont目录。默认只放了一种字体
2. 样本训练：Run with chinese_ocr.py。

### 字体切割&预测
1. Run with xuexin_segment.py。

### 训练需要的字体文件&训练好的模型下载

* 链接: https://pan.baidu.com/s/1h3pJ8UGQfCtfhNiyGA1NKg 提取码: nxyp

## 参考
* 训练相关参考：https://github.com/AstarLight/CPS-OCR-Engine