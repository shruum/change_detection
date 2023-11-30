from od.modeling.head.centernet.decode import centernet_decode


def centernet_eval_process(output, image_size, onnx_export):
    hm = output["hm"].sigmoid_()
    wh = output["wh"]
    reg = output["reg"]
    bboxes, clses, scores = centernet_decode(
        hm, wh, image_size, onnx_export, reg=reg, K=100
    )
    return bboxes, clses, scores, int(hm.size(3)), int(hm.size(2))
