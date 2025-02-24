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

        # 감지 상태 변수
        self.detection_start_time = None
        self.min_detection_duration = 1.0  # 1초 이상 감지 시 발행
        self.suppressed = False  # 1분 동안 감지 억제

        # Homography 행렬 (사용자 환경에 맞게 조정 필요)
        self.H = np.array([
            [-9.06094651e-03, -3.81663726e-02,  7.31503742e+00],
            [-1.71327616e-02,  4.45900570e-03, -7.27281751e-01],
            [ 1.87470875e-04,  7.00709590e-03,  1.00000000e+00]
        ])

    def image_callback(self, msg):
        if self.suppressed:  # 1분 억제 상태면 return
            return
        
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
                largest_contour = max(contours, key=cv2.conto
