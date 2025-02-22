#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

def publish_goal():
    rospy.init_node('custom_goal_publisher')
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

    # 카메라에서 받아오도록 바꾸기
    
    # Launch 파일에서 전달된 파라미터 읽기
    target_x = rospy.get_param('~target_x', 3.0)
    target_y = rospy.get_param('~target_y', 1.5)
    
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    
    # 위치 설정
    goal.pose.position.x = target_x
    goal.pose.position.y = target_y
    goal.pose.position.z = 0.0
    
    # 방향 설정 (정면 방향)
    q = quaternion_from_euler(0, 0, 0)
    goal.pose.orientation.x = q[0]
    goal.pose.orientation.y = q[1]
    goal.pose.orientation.z = q[2]
    goal.pose.orientation.w = q[3]
    
    # 5초 대기 (다른 노드 초기화 시간 확보)
    rospy.sleep(5)
    pub.publish(goal)
    rospy.loginfo(f"Published Goal: X={target_x}, Y={target_y}")

if __name__ == '__main__':
    try:
        publish_goal()
    except rospy.ROSInterruptException:
        pass
