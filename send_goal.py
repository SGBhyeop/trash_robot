#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from tf import TransformListener

class RedObjectDetector:
    def __init__(self):
        rospy.init_node('red_object_detector')
        
        # ROS 설정
        self.bridge = CvBridge()
        self.tf_listener = TransformListener()
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        # 이미지 서브스크라이버
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        
        # 빨간색 HSV 범위 (조정 필요)
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 감지 시간 변수
        self.detection_start_time = None
        self.min_detection_duration = 1.0  # 1초

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
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    # 1초 이상 감지 확인
                    if self.detection_start_time is None:
                        self.detection_start_time = rospy.Time.now()
                    else:
                        duration = (rospy.Time.now() - self.detection_start_time).to_sec()
                        if duration >= self.min_detection_duration:
                            self.publish_goal(cx, cy, cv_image.shape)
                            self.detection_start_time = None  # 재감지 방지
            else:
                self.detection_start_time = None

        except Exception as e:
            rospy.logerr(e)

    def publish_goal(self, pixel_x, pixel_y, image_shape):
        try:
            # 카메라 좌표 → 맵 좌표 변환
            (trans, rot) = self.tf_listener.lookupTransform(
                "map", 
                "camera_rgb_optical_frame",  # 카메라 TF 프레임 이름 확인 필수!
                rospy.Time(0)
            )
            
            # (픽셀 좌표 → 3D 좌표 변환 로직 추가 필요)
            # 여기서는 단순화된 예시 (실제 프로젝트에서는 depth 정보 사용)
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            
            # 임시 계산 (실제 환경에 맞게 조정)
            goal.pose.position.x = trans[0] + (pixel_x - image_shape[1]/2) * 0.001
            goal.pose.position.y = trans[1] + (image_shape[0]/2 - pixel_y) * 0.001
            goal.pose.orientation.w = 1.0  # 기본 방향
            
            self.goal_pub.publish(goal)
            rospy.loginfo(f"Published Goal: {goal.pose.position}")

        except Exception as e:
            rospy.logerr(f"TF Error: {e}")

if __name__ == '__main__':
    detector = RedObjectDetector()
    rospy.spin()
