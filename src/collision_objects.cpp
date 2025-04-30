#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("collision_objects_loader");

  moveit::planning_interface::PlanningSceneInterface psi;

  auto make_primitive = [](double x, double y, double z)
  {
    shape_msgs::msg::SolidPrimitive prim;
    prim.type = prim.BOX;
    prim.dimensions = {x, y, z};
    return prim;
  };
  auto make_pose = [](double x, double y, double z)
  {
    geometry_msgs::msg::Pose p;
    p.position.x = x;
    p.position.y = y;
    p.position.z = z;
    p.orientation.w = 1.0;
    return p;
  };

  std::vector<moveit_msgs::msg::CollisionObject> objs;
  objs.reserve(6);

  // ───── Wall ───────────────────────────────────────────────
  {
    moveit_msgs::msg::CollisionObject c;
    c.id = "Wall";
    c.header.frame_id = "base_link";
    c.primitives.push_back(make_primitive(1.2, 0.2, 1.2));
    c.primitive_poses.push_back(make_pose(0.0, -0.49, 0.5));
    c.operation = c.ADD;
    objs.push_back(c);
  }
  // ───── Table ───────────────────────────────────────────────
  {
    moveit_msgs::msg::CollisionObject c;
    c.id = "Table";
    c.header.frame_id = "base_link";
    c.primitives.push_back(make_primitive(1.0, 1.8, 0.15));
    c.primitive_poses.push_back(make_pose(0.38, 0.2, -0.08));
    c.operation = c.ADD;
    objs.push_back(c);
  }
  // ───── Coffee Machine ──────────────────────────────────────
  {
    moveit_msgs::msg::CollisionObject c;
    c.id = "Coffee_Machine";
    c.header.frame_id = "base_link";
    c.primitives.push_back(make_primitive(0.55, 0.7, 0.65));
    c.primitive_poses.push_back(make_pose(0.4, 0.85, 0.2));
    c.operation = c.ADD;
    objs.push_back(c);
  }
  // ───── Camera Zone ─────────────────────────────────────────
  {
    moveit_msgs::msg::CollisionObject c;
    c.id = "Camera_zone";
    c.header.frame_id = "base_link";
    c.primitives.push_back(make_primitive(0.50, 0.20, 0.20));
    c.primitive_poses.push_back(make_pose(-0.45, -0.40, 0.45));
    c.operation = c.ADD;
    objs.push_back(c);
  }
  // ───── Dispenser ───────────────────────────────────────────
  {
    moveit_msgs::msg::CollisionObject c;
    c.id = "Dispenser";
    c.header.frame_id = "base_link";
    c.primitives.push_back(make_primitive(0.19, 0.19, 0.45));
    c.primitive_poses.push_back(make_pose(0.28, -0.37, 0.55));
    c.operation = c.ADD;
    objs.push_back(c);
  }
  // ───── C-Clamp ─────────────────────────────────────────────
  {
    moveit_msgs::msg::CollisionObject c;
    c.id = "C-Clamp";
    c.header.frame_id = "base_link";
    c.primitives.push_back(make_primitive(0.16, 0.16, 0.20));
    c.primitive_poses.push_back(make_pose(-0.06, 0.25, -0.05));
    c.operation = c.ADD;
    objs.push_back(c);
  }

  psi.applyCollisionObjects(objs);
  RCLCPP_INFO(node->get_logger(), "✅ Collision objects published (6 objects)");

  rclcpp::sleep_for(std::chrono::seconds(2));
  rclcpp::shutdown();
  return 0;
}
