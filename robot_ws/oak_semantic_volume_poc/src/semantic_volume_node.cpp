#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "builtin_interfaces/msg/time.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"


namespace oak_semantic_volume_poc
{

struct Vec3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};


struct Key
{
  int x{0};
  int y{0};
  int z{0};

  bool operator==(const Key & other) const
  {
    return x == other.x &&
           y == other.y &&
           z == other.z;
  }
};


struct KeyHash
{
  std::size_t operator()(const Key & key) const noexcept
  {
    const std::size_t h1 = std::hash<int>{}(key.x);
    const std::size_t h2 = std::hash<int>{}(key.y);
    const std::size_t h3 = std::hash<int>{}(key.z);

    return h1 ^
           (h2 + 0x9e3779b9U + (h1 << 6U) + (h1 >> 2U)) ^
           (h3 + 0x9e3779b9U + (h2 << 6U) + (h2 >> 2U));
  }
};


struct SemanticCell
{
  int grid_x{0};
  int grid_y{0};

  int pixel_u{0};
  int pixel_v{0};

  std::uint16_t class_id{0};
};


struct MaskRegion
{
  int id{0};
  std::uint16_t class_id{0};

  std::vector<std::pair<int, int>> cells;
};


struct RayHit
{
  Key key;

  std::uint16_t class_id{0};

  int region_id{0};

  double range{0.0};
};


class SemanticVolumeNode : public rclcpp::Node
{
public:
  SemanticVolumeNode()
  : Node("semantic_volume_node"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    cloud_topic_ = declare_parameter<std::string>(
      "cloud_topic",
      "/livox/lidar");

    class_map_topic_ = declare_parameter<std::string>(
      "class_map_topic",
      "/oak_front/segmentation/class_map");

    camera_info_topic_ = declare_parameter<std::string>(
      "camera_info_topic",
      "/oak_front/rgb/camera_info");

    output_topic_ = declare_parameter<std::string>(
      "output_topic",
      "/semantic_volume/occupied_voxels");

    debug_hits_topic_ = declare_parameter<std::string>(
      "debug_hits_topic",
      "/semantic_volume/all_first_hits");

    target_frame_ = declare_parameter<std::string>(
      "target_frame",
      "body_link");

    camera_frame_parameter_ = declare_parameter<std::string>(
      "camera_frame",
      "oak_front_camera_optical_frame");

    voxel_size_ = declare_parameter<double>(
      "voxel_size_m",
      0.10);

    min_range_ = declare_parameter<double>(
      "min_range_m",
      0.35);

    max_range_ = declare_parameter<double>(
      "max_range_m",
      8.0);

    mask_cell_size_px_ =
      static_cast<int>(
          declare_parameter(
              "mask_cell_size_pixels",
              6L));

    mask_cell_size_px_ =
        std::max(mask_cell_size_px_,1);

    minimum_cell_class_fraction_ =
      declare_parameter<double>(
      "minimum_cell_class_fraction",
      0.60);

    erosion_px_ = std::max(
      0,
      static_cast<int>(declare_parameter<int>(
        "mask_erosion_pixels",
        2)));

    min_class_id_ = std::max(
      0,
      static_cast<int>(declare_parameter<int>(
        "minimum_class_id",
        1)));

    minimum_region_cells_ = std::max(
      1,
      static_cast<int>(declare_parameter<int>(
        "minimum_region_cells",
        4)));

    max_rays_ = std::max(
      1,
      static_cast<int>(declare_parameter<int>(
        "maximum_rays_per_frame",
        2500)));

    connectivity_radius_voxels_ = std::max(
      1,
      static_cast<int>(declare_parameter<int>(
        "hit_connectivity_radius_voxels",
        1)));

    minimum_component_hits_ = std::max(
      1,
      static_cast<int>(declare_parameter<int>(
        "minimum_component_hits",
        5)));

    minimum_hit_coverage_ =
      declare_parameter<double>(
      "minimum_hit_coverage",
      0.25);

    minimum_dominant_fraction_ =
      declare_parameter<double>(
      "minimum_dominant_fraction",
      0.70);

    publish_every_n_ = std::max(
      1,
      static_cast<int>(declare_parameter<int>(
        "publish_every_n_frames",
        1)));

    tf_timeout_sec_ = declare_parameter<double>(
      "tf_timeout_sec",
      0.05);

    if (voxel_size_ <= 0.0) {
      throw std::runtime_error(
              "voxel_size_m must be greater than zero.");
    }

    if (max_range_ <= min_range_) {
      throw std::runtime_error(
              "max_range_m must be greater than min_range_m.");
    }

    minimum_cell_class_fraction_ =
      std::max(
      0.0,
      std::min(1.0, minimum_cell_class_fraction_));

    minimum_hit_coverage_ =
      std::max(
      0.0,
      std::min(1.0, minimum_hit_coverage_));

    minimum_dominant_fraction_ =
      std::max(
      0.0,
      std::min(1.0, minimum_dominant_fraction_));

    const auto sensor_qos =
      rclcpp::SensorDataQoS().keep_last(1);

    cloud_sub_ =
      create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_,
      sensor_qos,
      std::bind(
        &SemanticVolumeNode::cloudCallback,
        this,
        std::placeholders::_1));

    camera_info_sub_ =
      create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      sensor_qos,
      std::bind(
        &SemanticVolumeNode::cameraInfoCallback,
        this,
        std::placeholders::_1));

    class_map_sub_ =
      create_subscription<sensor_msgs::msg::Image>(
      class_map_topic_,
      sensor_qos,
      std::bind(
        &SemanticVolumeNode::maskCallback,
        this,
        std::placeholders::_1));

    output_pub_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_,
      rclcpp::QoS(1));

    debug_pub_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
      debug_hits_topic_,
      rclcpp::QoS(1));

    RCLCPP_INFO(
      get_logger(),
      "Semantic viewing-volume node ready. "
      "Voxel: %.3f m, cell: %d px, max rays: %d, "
      "minimum dominant fraction: %.2f",
      voxel_size_,
      mask_cell_size_px_,
      max_rays_,
      minimum_dominant_fraction_);
  }


private:
  // ---------------------------------------------------------------------------
  // Camera information
  // ---------------------------------------------------------------------------

  void cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    latest_camera_info_ = *msg;
    have_camera_info_ = true;
  }


  // ---------------------------------------------------------------------------
  // LiDAR occupancy generation
  // ---------------------------------------------------------------------------

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (msg->width == 0 || msg->height == 0) {
      return;
    }

    geometry_msgs::msg::TransformStamped transform_msg;

    try {
      transform_msg = tf_buffer_.lookupTransform(
        target_frame_,
        msg->header.frame_id,
        rclcpp::Time(msg->header.stamp),
        rclcpp::Duration::from_seconds(tf_timeout_sec_));
    } catch (const tf2::TransformException & exception) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Could not transform cloud from '%s' to '%s': %s",
        msg->header.frame_id.c_str(),
        target_frame_.c_str(),
        exception.what());

      return;
    }

    tf2::Transform target_from_cloud;
    tf2::fromMsg(
      transform_msg.transform,
      target_from_cloud);

    std::unordered_set<Key, KeyHash> new_occupancy;

    const std::size_t expected_points =
      static_cast<std::size_t>(msg->width) *
      static_cast<std::size_t>(msg->height);

    new_occupancy.reserve(expected_points / 2U);

    try {
      sensor_msgs::PointCloud2ConstIterator<float>
      iterator_x(*msg, "x");

      sensor_msgs::PointCloud2ConstIterator<float>
      iterator_y(*msg, "y");

      sensor_msgs::PointCloud2ConstIterator<float>
      iterator_z(*msg, "z");

      for (
        ;
        iterator_x != iterator_x.end();
        ++iterator_x,
        ++iterator_y,
        ++iterator_z)
      {
        const double x = static_cast<double>(*iterator_x);
        const double y = static_cast<double>(*iterator_y);
        const double z = static_cast<double>(*iterator_z);

        if (
          !std::isfinite(x) ||
          !std::isfinite(y) ||
          !std::isfinite(z))
        {
          continue;
        }

        const tf2::Vector3 point_cloud(x, y, z);

        const tf2::Vector3 point_target =
          target_from_cloud * point_cloud;

        const Key key = pointToKey(
          Vec3{
            point_target.x(),
            point_target.y(),
            point_target.z()});

        new_occupancy.insert(key);
      }
    } catch (const std::runtime_error & exception) {
      RCLCPP_ERROR(
        get_logger(),
        "PointCloud2 is missing required x/y/z fields: %s",
        exception.what());

      return;
    }

    {
      std::lock_guard<std::mutex> lock(data_mutex_);

      occupancy_ = std::move(new_occupancy);
      occupancy_stamp_ = msg->header.stamp;
      have_occupancy_ = true;
    }

    RCLCPP_DEBUG(
      get_logger(),
      "Updated occupancy with %zu occupied voxels.",
      occupancy_.size());
  }


  // ---------------------------------------------------------------------------
  // Segmentation mask processing
  // ---------------------------------------------------------------------------

  void maskCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const auto start_time =
      std::chrono::steady_clock::now();

    sensor_msgs::msg::CameraInfo camera_info;

    std::unordered_set<Key, KeyHash> occupancy;

    {
      std::lock_guard<std::mutex> lock(data_mutex_);

      if (!have_camera_info_) {
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for CameraInfo.");

        return;
      }

      if (!have_occupancy_ || occupancy_.empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for LiDAR occupancy.");

        return;
      }

      camera_info = latest_camera_info_;
      occupancy = occupancy_;
    }

    if (msg->width == 0 || msg->height == 0) {
      return;
    }

    std::vector<std::uint16_t> mask;

    if (!decodeClassMap(*msg, mask)) {
      return;
    }

    if (erosion_px_ > 0) {
      mask = erodeClassMap(
        mask,
        static_cast<int>(msg->width),
        static_cast<int>(msg->height),
        erosion_px_);
    }

    const auto semantic_cells =
      buildSemanticCells(
      mask,
      static_cast<int>(msg->width),
      static_cast<int>(msg->height));

    const auto regions =
      buildMaskRegions(semantic_cells);

    if (regions.empty()) {
      publishTo(
        output_pub_,
        {},
        msg->header.stamp);

      publishTo(
        debug_pub_,
        {},
        msg->header.stamp);

      return;
    }

    const std::string camera_frame =
      camera_frame_parameter_.empty() ?
      camera_info.header.frame_id :
      camera_frame_parameter_;

    geometry_msgs::msg::TransformStamped camera_transform_msg;

    try {
      camera_transform_msg = tf_buffer_.lookupTransform(
        target_frame_,
        camera_frame,
        rclcpp::Time(msg->header.stamp),
        rclcpp::Duration::from_seconds(tf_timeout_sec_));
    } catch (const tf2::TransformException & exception) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Could not transform camera frame '%s' to '%s': %s",
        camera_frame.c_str(),
        target_frame_.c_str(),
        exception.what());

      return;
    }

    tf2::Transform target_from_camera;
    tf2::fromMsg(
      camera_transform_msg.transform,
      target_from_camera);

    const tf2::Vector3 camera_origin_tf =
      target_from_camera.getOrigin();

    const Vec3 camera_origin{
      camera_origin_tf.x(),
      camera_origin_tf.y(),
      camera_origin_tf.z()};

    const tf2::Matrix3x3 camera_rotation =
      target_from_camera.getBasis();

    const double fx = camera_info.k[0];
    const double fy = camera_info.k[4];
    const double cx = camera_info.k[2];
    const double cy = camera_info.k[5];

    if (
      fx <= 0.0 ||
      fy <= 0.0 ||
      camera_info.width == 0 ||
      camera_info.height == 0)
    {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "CameraInfo contains invalid intrinsics.");

      return;
    }

    std::unordered_map<Key, std::uint16_t, KeyHash>
    accepted_voxels;

    std::unordered_map<Key, std::uint16_t, KeyHash>
    all_first_hits;

    accepted_voxels.reserve(
      static_cast<std::size_t>(max_rays_));

    all_first_hits.reserve(
      static_cast<std::size_t>(max_rays_));

    int total_rays = 0;
    int total_first_hits = 0;
    int accepted_regions = 0;

    for (const auto & region : regions) {
      if (total_rays >= max_rays_) {
        break;
      }

      std::vector<RayHit> region_hits;
      region_hits.reserve(region.cells.size());

      int region_rays = 0;

      for (const auto & pixel : region.cells) {
        if (total_rays >= max_rays_) {
          break;
        }

        const int mask_u = pixel.first;
        const int mask_v = pixel.second;

        // Scale the segmentation-map pixel into the CameraInfo image.
        const double camera_u =
          (static_cast<double>(mask_u) + 0.5) *
          static_cast<double>(camera_info.width) /
          static_cast<double>(msg->width);

        const double camera_v =
          (static_cast<double>(mask_v) + 0.5) *
          static_cast<double>(camera_info.height) /
          static_cast<double>(msg->height);

        tf2::Vector3 direction_camera(
          (camera_u - cx) / fx,
          (camera_v - cy) / fy,
          1.0);

        if (direction_camera.length2() <= 1.0e-12) {
          continue;
        }

        direction_camera.normalize();

        const tf2::Vector3 direction_target =
          camera_rotation * direction_camera;

        Vec3 ray_direction{
          direction_target.x(),
          direction_target.y(),
          direction_target.z()};

        Key hit_key;
        double hit_range = 0.0;

        if (
          firstHit(
            camera_origin,
            ray_direction,
            occupancy,
            hit_key,
            hit_range))
        {
          region_hits.push_back(
            RayHit{
              hit_key,
              region.class_id,
              region.id,
              hit_range});

          all_first_hits[hit_key] =
            region.class_id;

          ++total_first_hits;
        }

        ++region_rays;
        ++total_rays;
      }

      std::vector<RayHit> accepted_hits;

      double hit_coverage = 0.0;
      double dominant_fraction = 0.0;

      const bool accepted =
        selectDominantComponent(
        region_hits,
        region_rays,
        accepted_hits,
        hit_coverage,
        dominant_fraction);

      if (!accepted) {
        RCLCPP_DEBUG(
          get_logger(),
          "Rejected region=%d class=%u rays=%d hits=%zu "
          "coverage=%.3f dominant=%.3f",
          region.id,
          static_cast<unsigned int>(region.class_id),
          region_rays,
          region_hits.size(),
          hit_coverage,
          dominant_fraction);

        continue;
      }

      for (const auto & hit : accepted_hits) {
        accepted_voxels[hit.key] =
          region.class_id;
      }

      ++accepted_regions;

      RCLCPP_DEBUG(
        get_logger(),
        "Accepted region=%d class=%u rays=%d hits=%zu "
        "coverage=%.3f dominant=%.3f accepted_hits=%zu",
        region.id,
        static_cast<unsigned int>(region.class_id),
        region_rays,
        region_hits.size(),
        hit_coverage,
        dominant_fraction,
        accepted_hits.size());
    }

    if ((frame_counter_++ % publish_every_n_) == 0) {
      publishTo(
        output_pub_,
        accepted_voxels,
        msg->header.stamp);

      publishTo(
        debug_pub_,
        all_first_hits,
        msg->header.stamp);
    }

    const auto end_time =
      std::chrono::steady_clock::now();

    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
      end_time - start_time).count();

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "regions=%zu accepted_regions=%d rays=%d "
      "first_hits=%d accepted_voxels=%zu "
      "geometry=%.2f ms",
      regions.size(),
      accepted_regions,
      total_rays,
      total_first_hits,
      accepted_voxels.size(),
      elapsed_ms);
  }


  // ---------------------------------------------------------------------------
  // Image decoding
  // ---------------------------------------------------------------------------

  bool decodeClassMap(
    const sensor_msgs::msg::Image & msg,
    std::vector<std::uint16_t> & output) const
  {
    const std::size_t pixel_count =
      static_cast<std::size_t>(msg.width) *
      static_cast<std::size_t>(msg.height);

    output.assign(pixel_count, 0U);

    if (
      msg.encoding == "mono16" ||
      msg.encoding == "16UC1")
    {
      const std::size_t required_row_bytes =
        static_cast<std::size_t>(msg.width) *
        sizeof(std::uint16_t);

      if (msg.step < required_row_bytes) {
        RCLCPP_ERROR(
          get_logger(),
          "Class-map step is too small for mono16/16UC1.");

        return false;
      }

      for (std::size_t y = 0; y < msg.height; ++y) {
        const std::uint8_t * row =
          msg.data.data() +
          y * static_cast<std::size_t>(msg.step);

        for (std::size_t x = 0; x < msg.width; ++x) {
          std::uint16_t value = 0U;

          std::memcpy(
            &value,
            row + x * sizeof(std::uint16_t),
            sizeof(std::uint16_t));

          if (msg.is_bigendian) {
            value = static_cast<std::uint16_t>(
              (value >> 8U) |
              (value << 8U));
          }

          output[y * msg.width + x] = value;
        }
      }

      return true;
    }

    if (
      msg.encoding == "mono8" ||
      msg.encoding == "8UC1")
    {
      if (msg.step < msg.width) {
        RCLCPP_ERROR(
          get_logger(),
          "Class-map step is too small for mono8/8UC1.");

        return false;
      }

      for (std::size_t y = 0; y < msg.height; ++y) {
        const std::uint8_t * row =
          msg.data.data() +
          y * static_cast<std::size_t>(msg.step);

        for (std::size_t x = 0; x < msg.width; ++x) {
          output[y * msg.width + x] =
            static_cast<std::uint16_t>(row[x]);
        }
      }

      return true;
    }

    RCLCPP_ERROR(
      get_logger(),
      "Unsupported class-map encoding '%s'. "
      "Expected mono8, 8UC1, mono16, or 16UC1.",
      msg.encoding.c_str());

    return false;
  }


  std::vector<std::uint16_t> erodeClassMap(
    const std::vector<std::uint16_t> & input,
    int width,
    int height,
    int radius) const
  {
    if (radius <= 0) {
      return input;
    }

    std::vector<std::uint16_t> output(
      input.size(),
      0U);

    for (int y = radius; y < height - radius; ++y) {
      for (int x = radius; x < width - radius; ++x) {
        const std::uint16_t class_id =
          input[
          static_cast<std::size_t>(y) *
          static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x)];

        if (
          class_id <
          static_cast<std::uint16_t>(min_class_id_))
        {
          continue;
        }

        bool keep = true;

        for (int dy = -radius; dy <= radius && keep; ++dy) {
          for (int dx = -radius; dx <= radius; ++dx) {
            const std::uint16_t neighbor =
              input[
              static_cast<std::size_t>(y + dy) *
              static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x + dx)];

            if (neighbor != class_id) {
              keep = false;
              break;
            }
          }
        }

        if (keep) {
          output[
            static_cast<std::size_t>(y) *
            static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)] =
            class_id;
        }
      }
    }

    return output;
  }


  // ---------------------------------------------------------------------------
  // Semantic-cell and region extraction
  // ---------------------------------------------------------------------------

  std::vector<SemanticCell> buildSemanticCells(
    const std::vector<std::uint16_t> & mask,
    int width,
    int height) const
  {
    std::vector<SemanticCell> cells;

    const int grid_width =
      (width + mask_cell_size_px_ - 1) /
      mask_cell_size_px_;

    const int grid_height =
      (height + mask_cell_size_px_ - 1) /
      mask_cell_size_px_;

    cells.reserve(
      static_cast<std::size_t>(grid_width) *
      static_cast<std::size_t>(grid_height) /
      4U);

    for (int grid_y = 0; grid_y < grid_height; ++grid_y) {
      for (int grid_x = 0; grid_x < grid_width; ++grid_x) {
        const int x0 =
          grid_x * mask_cell_size_px_;

        const int y0 =
          grid_y * mask_cell_size_px_;

        const int x1 =
          std::min(
          x0 + mask_cell_size_px_,
          width);

        const int y1 =
          std::min(
          y0 + mask_cell_size_px_,
          height);

        std::unordered_map<std::uint16_t, int>
        class_counts;

        int total_pixels = 0;

        for (int y = y0; y < y1; ++y) {
          for (int x = x0; x < x1; ++x) {
            const std::uint16_t class_id =
              mask[
              static_cast<std::size_t>(y) *
              static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x)];

            if (
              class_id >=
              static_cast<std::uint16_t>(min_class_id_))
            {
              ++class_counts[class_id];
            }

            ++total_pixels;
          }
        }

        std::uint16_t dominant_class = 0U;
        int dominant_count = 0;

        for (const auto & entry : class_counts) {
          if (entry.second > dominant_count) {
            dominant_class = entry.first;
            dominant_count = entry.second;
          }
        }

        if (
          dominant_class <
          static_cast<std::uint16_t>(min_class_id_) ||
          total_pixels <= 0)
        {
          continue;
        }

        const double class_fraction =
          static_cast<double>(dominant_count) /
          static_cast<double>(total_pixels);

        if (
          class_fraction <
          minimum_cell_class_fraction_)
        {
          continue;
        }

        cells.push_back(
          SemanticCell{
            grid_x,
            grid_y,
            (x0 + x1 - 1) / 2,
            (y0 + y1 - 1) / 2,
            dominant_class});
      }
    }

    return cells;
  }


  std::vector<MaskRegion> buildMaskRegions(
    const std::vector<SemanticCell> & cells) const
  {
    std::vector<MaskRegion> regions;

    if (cells.empty()) {
      return regions;
    }

    std::unordered_map<std::int64_t, int>
    cell_lookup;

    cell_lookup.reserve(cells.size());

    for (std::size_t index = 0; index < cells.size(); ++index) {
      cell_lookup[
        encodeGridCoordinate(
          cells[index].grid_x,
          cells[index].grid_y)] =
        static_cast<int>(index);
    }

    std::vector<bool> visited(
      cells.size(),
      false);

    const int neighbor_dx[4] =
    {
      1,
      -1,
      0,
      0
    };

    const int neighbor_dy[4] =
    {
      0,
      0,
      1,
      -1
    };

    int next_region_id = 0;

    for (
      std::size_t seed = 0;
      seed < cells.size();
      ++seed)
    {
      if (visited[seed]) {
        continue;
      }

      const std::uint16_t class_id =
        cells[seed].class_id;

      std::queue<int> pending;
      std::vector<int> component_indices;

      pending.push(
        static_cast<int>(seed));

      visited[seed] = true;

      while (!pending.empty()) {
        const int current =
          pending.front();

        pending.pop();

        component_indices.push_back(current);

        for (int neighbor_index = 0;
          neighbor_index < 4;
          ++neighbor_index)
        {
          const int neighbor_grid_x =
            cells[current].grid_x +
            neighbor_dx[neighbor_index];

          const int neighbor_grid_y =
            cells[current].grid_y +
            neighbor_dy[neighbor_index];

          const auto lookup_it =
            cell_lookup.find(
            encodeGridCoordinate(
              neighbor_grid_x,
              neighbor_grid_y));

          if (lookup_it == cell_lookup.end()) {
            continue;
          }

          const int neighbor =
            lookup_it->second;

          if (visited[neighbor]) {
            continue;
          }

          if (
            cells[neighbor].class_id !=
            class_id)
          {
            continue;
          }

          visited[neighbor] = true;
          pending.push(neighbor);
        }
      }

      if (
        static_cast<int>(component_indices.size()) <
        minimum_region_cells_)
      {
        continue;
      }

      MaskRegion region;

      region.id = next_region_id++;
      region.class_id = class_id;

      region.cells.reserve(
        component_indices.size());

      for (const int index : component_indices) {
        region.cells.emplace_back(
          cells[index].pixel_u,
          cells[index].pixel_v);
      }

      regions.push_back(
        std::move(region));
    }

    return regions;
  }


  // ---------------------------------------------------------------------------
  // Ray traversal
  // ---------------------------------------------------------------------------

  bool firstHit(
    const Vec3 & origin,
    Vec3 direction,
    const std::unordered_set<Key, KeyHash> & occupancy,
    Key & hit_key,
    double & hit_range) const
  {
    const double norm = std::sqrt(
      direction.x * direction.x +
      direction.y * direction.y +
      direction.z * direction.z);

    if (norm <= 1.0e-12) {
      return false;
    }

    direction.x /= norm;
    direction.y /= norm;
    direction.z /= norm;

    const Vec3 start{
      origin.x + min_range_ * direction.x,
      origin.y + min_range_ * direction.y,
      origin.z + min_range_ * direction.z};

    Key current = pointToKey(start);

    const int step_x =
      direction.x > 0.0 ? 1 :
      direction.x < 0.0 ? -1 : 0;

    const int step_y =
      direction.y > 0.0 ? 1 :
      direction.y < 0.0 ? -1 : 0;

    const int step_z =
      direction.z > 0.0 ? 1 :
      direction.z < 0.0 ? -1 : 0;

    const double infinity =
      std::numeric_limits<double>::infinity();

    const double t_delta_x =
      step_x == 0 ?
      infinity :
      voxel_size_ / std::abs(direction.x);

    const double t_delta_y =
      step_y == 0 ?
      infinity :
      voxel_size_ / std::abs(direction.y);

    const double t_delta_z =
      step_z == 0 ?
      infinity :
      voxel_size_ / std::abs(direction.z);

    const double next_boundary_x =
      step_x > 0 ?
      static_cast<double>(current.x + 1) *
      voxel_size_ :
      static_cast<double>(current.x) *
      voxel_size_;

    const double next_boundary_y =
      step_y > 0 ?
      static_cast<double>(current.y + 1) *
      voxel_size_ :
      static_cast<double>(current.y) *
      voxel_size_;

    const double next_boundary_z =
      step_z > 0 ?
      static_cast<double>(current.z + 1) *
      voxel_size_ :
      static_cast<double>(current.z) *
      voxel_size_;

    double t_max_x =
      step_x == 0 ?
      infinity :
      (next_boundary_x - start.x) /
      direction.x;

    double t_max_y =
      step_y == 0 ?
      infinity :
      (next_boundary_y - start.y) /
      direction.y;

    double t_max_z =
      step_z == 0 ?
      infinity :
      (next_boundary_z - start.z) /
      direction.z;

    t_max_x = std::max(0.0, t_max_x);
    t_max_y = std::max(0.0, t_max_y);
    t_max_z = std::max(0.0, t_max_z);

    double traveled_from_start = 0.0;

    const double maximum_travel =
      max_range_ - min_range_;

    constexpr double tie_epsilon = 1.0e-9;

    while (traveled_from_start <= maximum_travel) {
      if (
        occupancy.find(current) !=
        occupancy.end())
      {
        hit_key = current;

        hit_range =
          min_range_ +
          traveled_from_start;

        return true;
      }

      const double next_t =
        std::min(
        t_max_x,
        std::min(t_max_y, t_max_z));

      if (
        !std::isfinite(next_t) ||
        next_t > maximum_travel)
      {
        break;
      }

      if (
        std::abs(t_max_x - next_t) <=
        tie_epsilon)
      {
        current.x += step_x;
        t_max_x += t_delta_x;
      }

      if (
        std::abs(t_max_y - next_t) <=
        tie_epsilon)
      {
        current.y += step_y;
        t_max_y += t_delta_y;
      }

      if (
        std::abs(t_max_z - next_t) <=
        tie_epsilon)
      {
        current.z += step_z;
        t_max_z += t_delta_z;
      }

      traveled_from_start = next_t;
    }

    return false;
  }


  // ---------------------------------------------------------------------------
  // First-hit component grouping and dominant voting
  // ---------------------------------------------------------------------------

  std::vector<std::vector<int>> groupHitComponents(
    const std::vector<RayHit> & hits) const
  {
    std::vector<std::vector<int>> components;

    if (hits.empty()) {
      return components;
    }

    std::unordered_map<
      Key,
      std::vector<int>,
      KeyHash>
    hit_lookup;

    hit_lookup.reserve(hits.size());

    for (std::size_t index = 0; index < hits.size(); ++index) {
      hit_lookup[hits[index].key].push_back(
        static_cast<int>(index));
    }

    std::vector<bool> visited(
      hits.size(),
      false);

    for (
      std::size_t seed = 0;
      seed < hits.size();
      ++seed)
    {
      if (visited[seed]) {
        continue;
      }

      std::queue<int> pending;
      std::vector<int> component;

      pending.push(
        static_cast<int>(seed));

      visited[seed] = true;

      while (!pending.empty()) {
        const int current_index =
          pending.front();

        pending.pop();

        component.push_back(current_index);

        const Key current_key =
          hits[current_index].key;

        for (
          int dz = -connectivity_radius_voxels_;
          dz <= connectivity_radius_voxels_;
          ++dz)
        {
          for (
            int dy = -connectivity_radius_voxels_;
            dy <= connectivity_radius_voxels_;
            ++dy)
          {
            for (
              int dx = -connectivity_radius_voxels_;
              dx <= connectivity_radius_voxels_;
              ++dx)
            {
              const Key neighbor_key{
                current_key.x + dx,
                current_key.y + dy,
                current_key.z + dz};

              const auto lookup_it =
                hit_lookup.find(neighbor_key);

              if (lookup_it == hit_lookup.end()) {
                continue;
              }

              for (
                const int neighbor_index :
                lookup_it->second)
              {
                if (!visited[neighbor_index]) {
                  visited[neighbor_index] = true;
                  pending.push(neighbor_index);
                }
              }
            }
          }
        }
      }

      components.push_back(
        std::move(component));
    }

    return components;
  }


  bool selectDominantComponent(
    const std::vector<RayHit> & hits,
    int rays_cast,
    std::vector<RayHit> & accepted_hits,
    double & hit_coverage,
    double & dominant_fraction) const
  {
    accepted_hits.clear();

    hit_coverage = 0.0;
    dominant_fraction = 0.0;

    if (
      rays_cast <= 0 ||
      hits.empty())
    {
      return false;
    }

    hit_coverage =
      static_cast<double>(hits.size()) /
      static_cast<double>(rays_cast);

    if (
      hit_coverage <
      minimum_hit_coverage_)
    {
      return false;
    }

    const auto components =
      groupHitComponents(hits);

    if (components.empty()) {
      return false;
    }

    const std::vector<int> * dominant_component =
      nullptr;

    for (const auto & component : components) {
      if (
        dominant_component == nullptr ||
        component.size() >
        dominant_component->size())
      {
        dominant_component = &component;
      }
    }

    if (dominant_component == nullptr) {
      return false;
    }

    if (
      static_cast<int>(
        dominant_component->size()) <
      minimum_component_hits_)
    {
      return false;
    }

    dominant_fraction =
      static_cast<double>(
      dominant_component->size()) /
      static_cast<double>(hits.size());

    if (
      dominant_fraction <
      minimum_dominant_fraction_)
    {
      return false;
    }

    accepted_hits.reserve(
      dominant_component->size());

    for (
      const int hit_index :
      *dominant_component)
    {
      accepted_hits.push_back(
        hits[hit_index]);
    }

    return true;
  }


  // ---------------------------------------------------------------------------
  // Point cloud publishing
  // ---------------------------------------------------------------------------

  void publishTo(
    const rclcpp::Publisher<
      sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
    const std::unordered_map<
      Key,
      std::uint16_t,
      KeyHash> & labeled,
    const builtin_interfaces::msg::Time & stamp)
  {
    sensor_msgs::msg::PointCloud2 output;

    output.header.frame_id = target_frame_;
    output.header.stamp = stamp;

    output.height = 1;
    output.width =
      static_cast<std::uint32_t>(labeled.size());

    output.is_dense = true;
    output.is_bigendian = false;

    sensor_msgs::PointCloud2Modifier modifier(output);

    modifier.setPointCloud2Fields(
      4,
      "x",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "y",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "z",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "class_id",
      1,
      sensor_msgs::msg::PointField::UINT16);

    modifier.resize(labeled.size());

    sensor_msgs::PointCloud2Iterator<float>
    iterator_x(output, "x");

    sensor_msgs::PointCloud2Iterator<float>
    iterator_y(output, "y");

    sensor_msgs::PointCloud2Iterator<float>
    iterator_z(output, "z");

    sensor_msgs::PointCloud2Iterator<std::uint16_t>
    iterator_class(output, "class_id");

    for (const auto & entry : labeled) {
      const Vec3 point =
        keyCenter(entry.first);

      *iterator_x =
        static_cast<float>(point.x);

      *iterator_y =
        static_cast<float>(point.y);

      *iterator_z =
        static_cast<float>(point.z);

      *iterator_class =
        entry.second;

      ++iterator_x;
      ++iterator_y;
      ++iterator_z;
      ++iterator_class;
    }

    publisher->publish(output);
  }


  // ---------------------------------------------------------------------------
  // Coordinate helpers
  // ---------------------------------------------------------------------------

  Key pointToKey(
    const Vec3 & point) const
  {
    return Key{
      static_cast<int>(
        std::floor(point.x / voxel_size_)),
      static_cast<int>(
        std::floor(point.y / voxel_size_)),
      static_cast<int>(
        std::floor(point.z / voxel_size_))};
  }


  Vec3 keyCenter(
    const Key & key) const
  {
    return Vec3{
      (static_cast<double>(key.x) + 0.5) *
      voxel_size_,
      (static_cast<double>(key.y) + 0.5) *
      voxel_size_,
      (static_cast<double>(key.z) + 0.5) *
      voxel_size_};
  }


  static std::int64_t encodeGridCoordinate(
    int grid_x,
    int grid_y)
  {
    return
      (static_cast<std::int64_t>(
        static_cast<std::uint32_t>(grid_x))
      << 32U) |
      static_cast<std::uint32_t>(grid_y);
  }


  // ---------------------------------------------------------------------------
  // ROS interfaces
  // ---------------------------------------------------------------------------

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr
  cloud_sub_;

  rclcpp::Subscription<
    sensor_msgs::msg::Image>::SharedPtr
  class_map_sub_;

  rclcpp::Subscription<
    sensor_msgs::msg::CameraInfo>::SharedPtr
  camera_info_sub_;

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr
  output_pub_;

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr
  debug_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;


  // ---------------------------------------------------------------------------
  // Cached data
  // ---------------------------------------------------------------------------

  std::mutex data_mutex_;

  sensor_msgs::msg::CameraInfo
  latest_camera_info_;

  std::unordered_set<Key, KeyHash>
  occupancy_;

  builtin_interfaces::msg::Time
  occupancy_stamp_;

  bool have_camera_info_{false};
  bool have_occupancy_{false};


  // ---------------------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------------------

  std::string cloud_topic_;
  std::string class_map_topic_;
  std::string camera_info_topic_;

  std::string output_topic_;
  std::string debug_hits_topic_;

  std::string target_frame_;
  std::string camera_frame_parameter_;

  double voxel_size_{0.10};

  double min_range_{0.35};
  double max_range_{8.0};

  int mask_cell_size_px_{6};
  double minimum_cell_class_fraction_{0.60};

  int erosion_px_{2};
  int min_class_id_{1};

  int minimum_region_cells_{4};
  int max_rays_{2500};

  int connectivity_radius_voxels_{1};
  int minimum_component_hits_{5};

  double minimum_hit_coverage_{0.25};
  double minimum_dominant_fraction_{0.70};

  int publish_every_n_{1};
  int frame_counter_{0};

  double tf_timeout_sec_{0.05};
};

}  // namespace oak_semantic_volume_poc


int main(
  int argc,
  char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<
      oak_semantic_volume_poc::SemanticVolumeNode>());

  rclcpp::shutdown();

  return 0;
}