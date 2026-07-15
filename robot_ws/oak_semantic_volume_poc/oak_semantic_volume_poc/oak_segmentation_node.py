#!/usr/bin/env python3

import os
from typing import Dict, Optional

import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class OakSegmentationNode(Node):
    """
    Runs TopFormer-S ADE20K semantic segmentation on the OAK-1 Myriad X.

    OAK-1:
      - captures RGB
      - resizes RGB to 512 x 512
      - runs TopFormer inference
      - produces a 64 x 64 INT32 ADE20K class map

    Host:
      - reads the class map
      - publishes ROS images and CameraInfo
      - optionally generates a visualization overlay

    The host does not perform neural-network inference or argmax.
    """

    def __init__(self) -> None:
        super().__init__("oak_segmentation_node")

        defaults = {
            # Device/model
            "blob_path": "",
            "mxid": "",

            # ROS interface
            "camera_frame": "oak_left_rgb_camera_optical_frame",
            "rgb_topic": "/oak_left/rgb/image_raw",
            "camera_info_topic": "/oak_left/rgb/camera_info",
            "class_map_topic": "/oak_left/segmentation/class_map",
            "overlay_topic": "/oak_left/segmentation/overlay",

            # TopFormer-S configuration
            "preview_width": 512,
            "preview_height": 512,
            "fps": 20.0,
            "output_width": 64,
            "output_height": 64,
            "output_layer": "",

            # Reserve class 0 for background/unknown in the ROS interface.
            # Native ADE20K labels are 0...149, so publish them as 1...150.
            "class_id_offset": 1,

            # Runtime behavior
            "publish_rgb": True,
            "publish_overlay": True,
            "overlay_alpha": 0.45,
            "queue_size": 4,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)

        self.p: Dict[str, object] = {
            name: self.get_parameter(name).value
            for name in defaults
        }

        self._validate_parameters()

        self.rgb_pub = self.create_publisher(
            Image,
            str(self.p["rgb_topic"]),
            2,
        )

        self.info_pub = self.create_publisher(
            CameraInfo,
            str(self.p["camera_info_topic"]),
            2,
        )

        self.mask_pub = self.create_publisher(
            Image,
            str(self.p["class_map_topic"]),
            2,
        )

        self.overlay_pub = self.create_publisher(
            Image,
            str(self.p["overlay_topic"]),
            2,
        )

        self.pipeline = self._create_pipeline()
        self.device = self._open_device(self.pipeline)

        queue_size = int(self.p["queue_size"])

        self.q_rgb = self.device.getOutputQueue(
            name="rgb",
            maxSize=queue_size,
            blocking=False,
        )

        self.q_nn = self.device.getOutputQueue(
            name="nn",
            maxSize=queue_size,
            blocking=False,
        )

        self.K = self._read_intrinsics()
        self.D = np.zeros(5, dtype=np.float64)

        self.pending_rgb: Dict[int, np.ndarray] = {}
        self.latest_rgb: Optional[np.ndarray] = None

        # Poll rapidly without blocking the ROS executor.
        self.timer = self.create_timer(0.001, self.poll)

        mxid = self.device.getDeviceInfo().getMxId()

        self.get_logger().info(
            "TopFormer-S ADE20K pipeline started on OAK-1. "
            f"MXID={mxid}, "
            f"input={self.p['preview_width']}x{self.p['preview_height']}, "
            f"output={self.p['output_width']}x{self.p['output_height']}"
        )

    def _validate_parameters(self) -> None:
        blob_path = str(self.p["blob_path"])

        if not os.path.isfile(blob_path):
            raise FileNotFoundError(
                "TopFormer blob not found. Set blob_path to a valid "
                f"Myriad-X .blob file. Received: '{blob_path}'"
            )

        preview_width = int(self.p["preview_width"])
        preview_height = int(self.p["preview_height"])
        output_width = int(self.p["output_width"])
        output_height = int(self.p["output_height"])

        if preview_width <= 0 or preview_height <= 0:
            raise ValueError("Preview dimensions must be positive.")

        if output_width <= 0 or output_height <= 0:
            raise ValueError("Output dimensions must be positive.")

        if preview_width != 512 or preview_height != 512:
            self.get_logger().warning(
                "The referenced TopFormer blob expects 512x512 input. "
                f"Configured input is {preview_width}x{preview_height}."
            )

        if output_width != 64 or output_height != 64:
            self.get_logger().warning(
                "The referenced TopFormer output is expected to be 64x64. "
                f"Configured output is {output_width}x{output_height}."
            )

    def _create_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        # The referenced blob was compiled for OpenVINO 2021.4.
        try:
            pipeline.setOpenVINOVersion(
                dai.OpenVINO.Version.VERSION_2021_4
            )
        except AttributeError:
            # Compatibility with DepthAI versions exposing the enum directly.
            try:
                pipeline.setOpenVINOVersion(
                    dai.OpenVINO.VERSION_2021_4
                )
            except AttributeError:
                self.get_logger().warning(
                    "Could not explicitly select OpenVINO 2021.4. "
                    "The blob may still specify the required version."
                )

        camera = pipeline.create(dai.node.ColorCamera)

        camera.setPreviewSize(
            int(self.p["preview_width"]),
            int(self.p["preview_height"]),
        )

        # Use a direct resize to 512x512. This keeps the class-map-to-image
        # coordinate relationship simple and deterministic.
        camera.setPreviewKeepAspectRatio(False)
        camera.setInterleaved(False)
        camera.setColorOrder(
            dai.ColorCameraProperties.ColorOrder.BGR
        )
        camera.setFps(float(self.p["fps"]))

        neural_network = pipeline.create(dai.node.NeuralNetwork)
        neural_network.setBlobPath(str(self.p["blob_path"]))
        neural_network.setNumInferenceThreads(2)
        neural_network.setNumPoolFrames(4)
        neural_network.input.setBlocking(False)
        neural_network.input.setQueueSize(2)

        camera.preview.link(neural_network.input)

        rgb_output = pipeline.create(dai.node.XLinkOut)
        rgb_output.setStreamName("rgb")

        nn_output = pipeline.create(dai.node.XLinkOut)
        nn_output.setStreamName("nn")

        # Publish exactly the image supplied to the neural network.
        neural_network.passthrough.link(rgb_output.input)
        neural_network.out.link(nn_output.input)

        return pipeline

    def _open_device(self, pipeline: dai.Pipeline) -> dai.Device:
        mxid = str(self.p["mxid"]).strip()

        if not mxid:
            self.get_logger().warning(
                "No MXID configured. Connecting to the next available "
                "DepthAI device."
            )
            return dai.Device(pipeline)

        device_info = dai.DeviceInfo(mxid)

        self.get_logger().info(
            f"Connecting to OAK device MXID={mxid}"
        )

        return dai.Device(pipeline, device_info)

    def _read_intrinsics(self) -> np.ndarray:
        calibration = self.device.readCalibration()

        width = int(self.p["preview_width"])
        height = int(self.p["preview_height"])

        try:
            intrinsics = calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.RGB,
                width,
                height,
            )
        except RuntimeError:
            # Some newer devices expose the color socket as CAM_A.
            intrinsics = calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A,
                width,
                height,
            )

        K = np.asarray(intrinsics, dtype=np.float64)

        if K.shape != (3, 3):
            raise RuntimeError(
                f"Unexpected intrinsic matrix shape: {K.shape}"
            )

        if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise RuntimeError(
                f"Invalid OAK camera intrinsics:\n{K}"
            )

        self.get_logger().info(
            "Camera intrinsics: "
            f"fx={K[0, 0]:.3f}, fy={K[1, 1]:.3f}, "
            f"cx={K[0, 2]:.3f}, cy={K[1, 2]:.3f}"
        )

        return K

    @staticmethod
    def image_msg(
        array: np.ndarray,
        encoding: str,
        stamp,
        frame_id: str,
    ) -> Image:
        contiguous = np.ascontiguousarray(array)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(contiguous.shape[0])
        msg.width = int(contiguous.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(contiguous.strides[0])
        msg.data = contiguous.tobytes()

        return msg

    def camera_info(self, stamp) -> CameraInfo:
        width = int(self.p["preview_width"])
        height = int(self.p["preview_height"])
        frame_id = str(self.p["camera_frame"])

        msg = CameraInfo()

        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.width = width
        msg.height = height

        msg.distortion_model = "plumb_bob"
        msg.d = self.D.tolist()
        msg.k = self.K.reshape(-1).tolist()
        msg.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()

        msg.p = [
            float(self.K[0, 0]),
            0.0,
            float(self.K[0, 2]),
            0.0,

            0.0,
            float(self.K[1, 1]),
            float(self.K[1, 2]),
            0.0,

            0.0,
            0.0,
            1.0,
            0.0,
        ]

        return msg

    def decode_class_map(self, packet: dai.NNData) -> np.ndarray:
        """
        Decode the TopFormer INT32 output.

        Expected native output:
            shape: 64 x 64
            values: ADE20K class IDs 0...149
        """

        layer_name = str(self.p["output_layer"]).strip()

        if layer_name:
            raw = np.asarray(
                packet.getLayerInt32(layer_name),
                dtype=np.int32,
            )
        else:
            raw = np.asarray(
                packet.getFirstLayerInt32(),
                dtype=np.int32,
            )

        output_width = int(self.p["output_width"])
        output_height = int(self.p["output_height"])

        expected_values = output_width * output_height

        if raw.size != expected_values:
            layer_names = packet.getAllLayerNames()

            raise RuntimeError(
                "Unexpected TopFormer output size. "
                f"Received {raw.size} values, expected {expected_values}. "
                f"Available layers: {layer_names}"
            )

        class_map = raw.reshape(
            output_height,
            output_width,
        )

        native_min = int(class_map.min())
        native_max = int(class_map.max())

        if native_min < 0 or native_max >= 150:
            self.get_logger().warning(
                "TopFormer class IDs are outside the expected ADE20K "
                f"range 0...149: min={native_min}, max={native_max}"
            )

        # Reserve ROS class ID 0 for background/unknown.
        offset = int(self.p["class_id_offset"])

        shifted = class_map.astype(np.int64) + offset

        shifted = np.clip(
            shifted,
            0,
            np.iinfo(np.uint16).max,
        )

        return shifted.astype(np.uint16)

    @staticmethod
    def colorize_class_map(
        class_map: np.ndarray,
    ) -> np.ndarray:
        """
        Create a deterministic debugging palette.

        This is only a visualization; the mono16 topic contains the real
        semantic class IDs.
        """

        ids = class_map.astype(np.uint32)

        blue = ((ids * 37) % 255).astype(np.uint8)
        green = ((ids * 67) % 255).astype(np.uint8)
        red = ((ids * 97) % 255).astype(np.uint8)

        color = np.stack(
            [blue, green, red],
            axis=-1,
        )

        # Class zero remains black.
        color[class_map == 0] = 0

        return color

    def poll(self) -> None:
        # Drain available RGB packets.
        while True:
            rgb_packet = self.q_rgb.tryGet()

            if rgb_packet is None:
                break

            sequence = int(rgb_packet.getSequenceNum())
            frame = rgb_packet.getCvFrame()

            self.pending_rgb[sequence] = frame
            self.latest_rgb = frame

        # Limit memory if NN packets are delayed.
        if len(self.pending_rgb) > 20:
            oldest_sequences = sorted(self.pending_rgb.keys())[:-10]

            for sequence in oldest_sequences:
                self.pending_rgb.pop(sequence, None)

        # Drain available NN packets.
        while True:
            nn_packet = self.q_nn.tryGet()

            if nn_packet is None:
                break

            sequence = int(nn_packet.getSequenceNum())

            rgb_frame = self.pending_rgb.pop(
                sequence,
                self.latest_rgb,
            )

            if rgb_frame is None:
                self.get_logger().warning(
                    "Received NN output without a corresponding RGB frame."
                )
                continue

            try:
                class_map = self.decode_class_map(nn_packet)
            except Exception as exc:
                self.get_logger().error(
                    f"TopFormer decode failed: {exc}"
                )
                continue

            stamp = self.get_clock().now().to_msg()
            frame_id = str(self.p["camera_frame"])

            if bool(self.p["publish_rgb"]):
                self.rgb_pub.publish(
                    self.image_msg(
                        rgb_frame,
                        "bgr8",
                        stamp,
                        frame_id,
                    )
                )

            self.info_pub.publish(
                self.camera_info(stamp)
            )

            # Publish the original 64x64 semantic map. The geometry node
            # scales mask coordinates to CameraInfo coordinates.
            self.mask_pub.publish(
                self.image_msg(
                    class_map,
                    "mono16",
                    stamp,
                    frame_id,
                )
            )

            if bool(self.p["publish_overlay"]):
                color_small = self.colorize_class_map(class_map)

                color_full = cv2.resize(
                    color_small,
                    (
                        int(self.p["preview_width"]),
                        int(self.p["preview_height"]),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                alpha = float(self.p["overlay_alpha"])
                alpha = max(0.0, min(1.0, alpha))

                overlay = cv2.addWeighted(
                    rgb_frame,
                    1.0 - alpha,
                    color_full,
                    alpha,
                    0.0,
                )

                self.overlay_pub.publish(
                    self.image_msg(
                        overlay,
                        "bgr8",
                        stamp,
                        frame_id,
                    )
                )

    def destroy_node(self):
        try:
            if hasattr(self, "device"):
                self.device.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node: Optional[OakSegmentationNode] = None

    try:
        node = OakSegmentationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print(f"oak_segmentation_node failed: {exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()