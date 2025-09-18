#!/usr/bin/env python3
"""Convert a YOLOv8 ONNX model to TFLite and sanity-check the result."""
import argparse
import os
import pathlib
import shutil
import sys
from typing import Optional, Sequence, Tuple

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import onnx
import onnxruntime as ort
from onnx2tf import convert
import tensorflow as tf


def parse_shape(shape_arg: str) -> Tuple[int, ...]:
    parts = [segment.strip() for segment in shape_arg.split(",") if segment.strip()]
    if not parts:
        raise ValueError("--input-shape must contain at least one dimension")
    return tuple(int(value) for value in parts)


def extract_input_info(onnx_path: pathlib.Path) -> Tuple[str, Sequence[Optional[int]]]:
    model = onnx.load(str(onnx_path))
    try:
        if not model.graph.input:
            raise RuntimeError("ONNX model exposes no inputs")
        tensor = model.graph.input[0]
        dims = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        return tensor.name, dims
    finally:
        del model


def make_static_shape(dims: Sequence[Optional[int]], override: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    if override is not None:
        return override
    resolved = []
    for idx, dim in enumerate(dims):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif isinstance(dim, str) and dim.isdigit():
            resolved.append(int(dim))
        else:
            raise ValueError(
                f"Dynamic dimension detected at index {idx}. "
                "Use --input-shape to supply a concrete size."
            )
    return tuple(resolved)


def prepare_output_dir(path: pathlib.Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_conversion(
    onnx_path: pathlib.Path,
    output_dir: pathlib.Path,
    overwrite_shape: Optional[Tuple[int, ...]],
    input_name: str,
) -> None:
    kwargs = dict(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    if overwrite_shape is not None:
        shape_str = ",".join(str(dim) for dim in overwrite_shape)
        kwargs["overwrite_input_shape"] = [f"{input_name}:{shape_str}"]
    convert(**kwargs)


def pick_float32_tflite(output_dir: pathlib.Path) -> pathlib.Path:
    candidates = sorted(output_dir.glob("*float32.tflite"))
    if not candidates:
        raise FileNotFoundError("onnx2tf did not emit a float32 .tflite artifact")
    return candidates[-1]


def verify_models(
    onnx_path: pathlib.Path,
    tflite_path: pathlib.Path,
    input_name: str,
    input_shape: Tuple[int, ...],
    samples: int,
    rtol: float,
    atol: float,
) -> None:
    np.random.seed(0)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    if len(input_details) != 1:
        raise RuntimeError(f"Expected a single input, found {len(input_details)}")
    tflite_shape = tuple(int(size) for size in input_details[0]["shape"])
    transpose_axes: Optional[Tuple[int, ...]] = None
    if input_shape != tflite_shape:
        if len(input_shape) == 4 and input_shape[0] == tflite_shape[0]:
            if (
                input_shape[1] == tflite_shape[3]
                and input_shape[2] == tflite_shape[1]
                and input_shape[3] == tflite_shape[2]
            ):
                transpose_axes = (0, 2, 3, 1)
            else:
                raise ValueError(
                    f"Unable to align ONNX input shape {input_shape} with TFLite shape {tflite_shape}"
                )
        else:
            raise ValueError(
                f"Unable to align ONNX input shape {input_shape} with TFLite shape {tflite_shape}"
            )
    interpreter.resize_tensor_input(input_details[0]["index"], tflite_shape, strict=True)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    if len(output_details) != 1:
        raise RuntimeError(f"Expected a single output, found {len(output_details)}")
    for sample_idx in range(samples):
        sample = np.random.rand(*input_shape).astype(np.float32)
        onnx_outputs = session.run(None, {input_name: sample})
        if len(onnx_outputs) != 1:
            raise RuntimeError(f"Expected one ONNX output, found {len(onnx_outputs)}")
        tflite_input = sample.transpose(transpose_axes) if transpose_axes else sample
        interpreter.set_tensor(input_details[0]["index"], tflite_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            onnx_outputs[0],
            tflite_output,
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch at sample {sample_idx}",
        )
    print(f"Verification passed on {samples} random inputs (rtol={rtol}, atol={atol}).")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=pathlib.Path,
        default=pathlib.Path("golf_ball_detector.onnx"),
        help="Path to the source ONNX model.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("artifacts"),
        help="Directory for conversion outputs.",
    )
    parser.add_argument(
        "--tflite-name",
        default="golf_ball_detector.tflite",
        help="Filename for the canonical float32 TFLite artifact.",
    )
    parser.add_argument(
        "--input-shape",
        type=parse_shape,
        help="Override input shape, e.g. 1,3,640,640.",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=5,
        help="Number of random samples for output comparison (0 to skip).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="Relative tolerance for verification.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for verification.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clear the output directory before conversion.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip output comparison between ONNX and TFLite.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    onnx_path = args.onnx.resolve()
    if not onnx_path.exists():
        parser.error(f"ONNX model not found at {onnx_path}")
    output_dir = args.out_dir.resolve()
    try:
        input_name, raw_dims = extract_input_info(onnx_path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to inspect ONNX model: {exc}") from exc
    try:
        input_shape = make_static_shape(raw_dims, args.input_shape)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Unable to resolve input shape: {exc}") from exc
    prepare_output_dir(output_dir, clean=not args.no_clean)
    print(f"Converting {onnx_path.name} -> {output_dir} ...")
    run_conversion(onnx_path, output_dir, args.input_shape, input_name)
    float32_src = pick_float32_tflite(output_dir)
    final_tflite = output_dir / args.tflite_name
    if float32_src.resolve() != final_tflite.resolve():
        if final_tflite.exists():
            final_tflite.unlink()
        shutil.move(str(float32_src), str(final_tflite))
    else:
        final_tflite = float32_src
    print(f"Float32 TFLite available at {final_tflite}")
    if not args.skip_verify and args.verify_samples > 0:
        verify_models(
            onnx_path=onnx_path,
            tflite_path=final_tflite,
            input_name=input_name,
            input_shape=input_shape,
            samples=args.verify_samples,
            rtol=args.rtol,
            atol=args.atol,
        )
    else:
        print("Verification skipped.")
    print("Done.")


if __name__ == "__main__":
    main()
