from face_detection import FaceDetector
from head_pose_estimation import HeadPoseEstimator
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimator


class ClassificationManager:

    def initialize_models(self):

        mfd = r'../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
        mfld = r'../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
        mhpe = r'../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
        mge = r'../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'
        device = 'CPU'

        print('Initializing Estimators')
        self.face_detector = FaceDetector(mfd, device)
        self.facial_landmarks_detector = FacialLandmarksDetector(mfld, device)
        self.head_pose_estimator = HeadPoseEstimator(mhpe, device)
        self.gaze_estimator = GazeEstimator(mge, device)

        print('Loading Models')
        self.face_detector.load_model()
        self.facial_landmarks_detector.load_model()
        self.head_pose_estimator.load_model()
        self.gaze_estimator.load_model()

    def predict_gaze_position(self, image):
        face_boxes = self.face_detector.predict(image)
        # face_box = face_boxes[0]
        # Obsługa dla braku twarzy w obrazie z kamery i kiedy > 1 - może wiąć największą ?
        largest_face_box = self.get_largest_face_box(face_boxes)
        face_image = self.get_crop_image(image, largest_face_box)
        eye_boxes, eye_centers = self.facial_landmarks_detector.predict(face_image)
        left_eye_image, right_eye_image = [self.get_crop_image(face_image, eye_box) for eye_box in eye_boxes]
        head_pose_angles = self.head_pose_estimator.predict(face_image)
        gaze_x, gaze_y = self.gaze_estimator.predict(right_eye_image, head_pose_angles, left_eye_image)
        return gaze_x, gaze_y

    def predict_face_position(self, image):
        face_boxes = self.face_detector.predict(image)
        largest_face_box = self.get_largest_face_box(face_boxes)
        return [largest_face_box]
        #return face_boxes

    def get_crop_image(self, image, box):
        xmin, ymin, xmax, ymax = box
        crop_image = image[ymin:ymax, xmin:xmax]
        return crop_image

    def get_largest_face_box(self, face_boxes):
        return max(face_boxes, key=lambda x: (x[2] - x[0]) + (x[3] - x[1]))