"""Export the trained U-Net for in-browser inference.

Run from the repository root (needs: pip install onnx onnxruntime onnxscript):
    python -m backend.export_onnx --weights backend/corrosion_unet_enhanced.pth

Writes public/models/corrosion_unet.onnx with int8 weights (about 30 MB), which the
dashboard downloads and runs with onnxruntime-web. Pass --fp32 to also keep the
full-precision export (about 120 MB) for comparison.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .model import UNet

SIZE = 256


def load(weights):
    model = UNet()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


def fold_batchnorm(model):
    # Each CBR block is Conv, BN, ReLU, Conv, BN, ReLU. Folding BN into the convs gives
    # identical outputs and lets the int8 step quantise the final weights directly.
    for block in [model.enc1, model.enc2, model.enc3, model.enc4, model.bridge, model.dec4, model.dec3, model.dec2, model.dec1]:
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = torch.nn.Identity()
        block[3] = fuse_conv_bn_eval(block[3], block[4])
        block[4] = torch.nn.Identity()
    return model


def export(model, path):
    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(model, (dummy,), str(path), input_names=["image"], output_names=["probability"], opset_version=18, dynamo=True, external_data=False, verbose=False)
    onnx.checker.check_model(str(path))


def quantize_weights(src, dst):
    """Store Conv/ConvTranspose weights as per-output-channel int8 plus a DequantizeLinear node.

    Activations stay float32, so every runtime that runs the float model runs this one;
    only the download shrinks about 4x.
    """
    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    new_nodes = []
    for node in graph.node:
        if node.op_type in ("Conv", "ConvTranspose") and node.input[1] in inits:
            name = node.input[1]
            w = numpy_helper.to_array(inits[name]).astype(np.float32)
            # Conv weights are [out, in, kh, kw]; ConvTranspose weights are [in, out, kh, kw].
            axis = 0 if node.op_type == "Conv" else 1
            reduce = tuple(i for i in range(w.ndim) if i != axis)
            scale = np.maximum(np.abs(w).max(axis=reduce) / 127.0, 1e-12).astype(np.float32)
            shape = [1] * w.ndim
            shape[axis] = -1
            q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
            graph.initializer.remove(inits[name])
            graph.initializer.extend([
                numpy_helper.from_array(q, name + "_q"),
                numpy_helper.from_array(scale, name + "_scale"),
            ])
            new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale"], [name], axis=axis, name=name + "_dequantize"))
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def compare(model, path, samples):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst_diff, worst_agree, worst_cov = 0.0, 1.0, 0.0
    for x in samples:
        with torch.inference_mode():
            ref = model(torch.from_numpy(x)).numpy()
        out = session.run(None, {"image": x})[0]
        worst_diff = max(worst_diff, float(np.abs(ref - out).max()))
        worst_agree = min(worst_agree, float(((ref > 0.5) == (out > 0.5)).mean()))
        worst_cov = max(worst_cov, abs(float((ref > 0.5).mean() - (out > 0.5).mean())) * 100)
    print(f"  {path.name}: max probability difference {worst_diff:.4f}, mask agreement {worst_agree * 100:.2f}%, coverage difference {worst_cov:.2f} points")
    return worst_agree


def sample_inputs(folder):
    if folder:
        import cv2
        from .model import enhance
        for p in sorted(Path(folder).glob("*.jpg"))[:20]:
            image = enhance(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            image = cv2.resize(image, (SIZE, SIZE))
            yield (image.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    else:
        rng = np.random.default_rng(0)
        for _ in range(4):
            yield rng.random((1, 3, SIZE, SIZE), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="backend/corrosion_unet_enhanced.pth")
    parser.add_argument("--out", default="public/models/corrosion_unet.onnx")
    parser.add_argument("--images", help="folder of validation .jpg photos to compare on (random inputs otherwise)")
    parser.add_argument("--fp32", action="store_true", help="also keep the full-precision export")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = out.with_name(out.stem + "_fp32.onnx")
    reference = load(args.weights)
    export(fold_batchnorm(load(args.weights)), fp32)
    quantize_weights(fp32, out)

    samples = list(sample_inputs(args.images))
    print("Checking exports against PyTorch:")
    compare(reference, fp32, samples)
    agreement = compare(reference, out, samples)
    if not args.fp32:
        fp32.unlink()
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if agreement < 0.99:
        print("Warning: int8 masks differ from PyTorch on more than 1% of pixels. Consider deploying the --fp32 export instead.")


if __name__ == "__main__":
    main()
