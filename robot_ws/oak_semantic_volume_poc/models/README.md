# Model contract

Default target: DeepLabV3 + MobileNetV2 trained on ADE20K, compiled for RVC2/Myriad X.
The repository intentionally does not redistribute weights or a blob.

Expected default output: FP16 logits, NCHW, `[1,150,80,128]` for a 512x320 input.
Update `config/poc.yaml` to the actual exported tensor dimensions and layer name.
For minimum USB bandwidth, compile/export a graph with ArgMax and set
`output_is_class_map: true`; then the node expects one HxW class-ID tensor.

A model used in OAK Viewer can replace this default. Copy its `.blob` locally and
match the output settings; the C++ geometry node requires only the `mono16` class map.
