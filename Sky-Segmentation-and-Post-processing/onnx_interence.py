import os
import copy
import argparse
import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv.cvtColor(resize_image, cv.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})[0]

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype('uint8')

    return onnx_result


def main():
    parser = argparse.ArgumentParser(description='U-2-Net sky segmentation inference (onnxruntime)')
    parser.add_argument('--image', type=str, required=True, help='input image path')
    parser.add_argument('--onnx', type=str, default='skyseg.onnx', help='onnx model path')
    parser.add_argument('--out', type=str, default=None, help='output mask path (default: <image>_sky_mask.png)')
    parser.add_argument('--size', type=int, nargs=2, default=[320, 320], help='model input size, e.g. 320 320')
    args = parser.parse_args()

    image = cv.imread(args.image)
    if image is None:
        print('cannot read image:', args.image)
        return

    # keep the input resolution reasonable (same as the official demo)
    while image.shape[0] >= 640 and image.shape[1] >= 640:
        image = cv.pyrDown(image)

    onnx_session = onnxruntime.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    result_map = run_inference(onnx_session, args.size, image)
    result_map = cv.resize(result_map, (image.shape[1], image.shape[0]), interpolation=cv.INTER_LINEAR)

    out_path = args.out
    if out_path is None:
        base, _ = os.path.splitext(args.image)
        out_path = base + '_sky_mask.png'
    cv.imwrite(out_path, result_map)
    print('saved:', out_path)


if __name__ == "__main__":
    main()
