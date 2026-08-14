from typing import TYPE_CHECKING

'''
paddleocr：解析图片中的文字，也可以进行表格识别
rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
主要区别在于它们所使用的推理引擎和硬件支持
选择哪种方式最合适取决于你的硬件环境和性能需求。
当你有 GPU 且追求速度时：使用 rapidocr_paddle。PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。它在 CPU 上进行了优化，资源占用较低.
'''


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        #
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr


if __name__ == '__main__':
    model = get_ocr()
    result, data = model(r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system/rag_qa/data/samples/ocr_04.png")  # 是传给RapidOCR里面call方法的
    for each in result:
        print(each)
        """
        [[坐标], '文本', 置信度]
        [[[234.0, 127.0], [462.0, 126.0], [462.0, 150.0], [234.0, 151.0]], 'public String getBookNameO', 0.9754688579302567]
        
        """