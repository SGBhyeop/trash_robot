#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class RedObjectDetector:
    def __init__(self):
        rospy.init_node('red_object_detector')

        # ROS 설정
        self.bridge = CvBridge()
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # 이미지 서브스크라이버
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        # 빨간색 HSV 범위
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        # 감지 시간 변수
        self.detection_start_time = None
        self.min_detection_duration = 1.0  # 1초 이상 감지 시 발행

        # Homography 행렬 설정
        self.H = np.array([[-9.06094651e-03, -3.81663726e-02,  7.31503742e+00],
       [-1.71327616e-02,  4.45900570e-03, -7.27281751e-01],
       [ 1.87470875e-04,  7.00709590e-03,  1.00000000e+00]])

    def image_callback(self, msg):
        try:
            # 이미지 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # 빨간색 마스크 생성
            mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # 물체 감지
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # 최소 면적 필터
                    M = cv2.moments(largest_contour)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # 1초 이상 감지 확인
                    if self.detection_start_time is None:
                        self.detection_start_time = rospy.Time.now()
                    else:
                        duration = (rospy.Time.now() - self.detection_start_time).to_sec()
                        if duration >= self.min_detection_duration:
                            self.publish_goal(cx, cy)
                            self.detection_start_time = None  # 재감지 방지
            else:
                self.detection_start_time = None

        except Exception as e:
            rospy.logerr(e)

    def publish_goal(self, pixel_x, pixel_y):
        try:
            # 픽셀 좌표 → 맵 좌표 변환 (Homography 이용)
            pixel_coord = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            world_coord = cv2.perspectiveTransform(pixel_coord, self.H)
            wx, wy = world_coord[0][0]

            # ROS 메시지 생성 및 발행
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = wx
            goal.pose.position.y = wy
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0  # 기본 방향

            self.goal_pub.publish(goal)
            rospy.loginfo(f"Published Goal: x={wx}, y={wy}")

        except Exception as e:
            rospy.logerr(f"Homography Transformation Error: {e}")

if __name__ == '__main__':
    detector = RedObjectDetector()
    rospy.spin()
