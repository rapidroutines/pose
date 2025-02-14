import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.counter = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point
        c = np.array(c)  # End point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def count_reps(self, angle, threshold_down, threshold_up):
        if angle > threshold_down:
            self.stage = "down"
        if angle < threshold_up and self.stage == "down":
            self.stage = "up"
            self.counter += 1

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = results.pose_landmarks

        return image, landmarks

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(
            image, landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    def run(self):
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue

            try:
                image, landmarks = self.process_frame(frame)

                if landmarks:
                    # Example for bicep curl
                    shoulder = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    elbow = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                             landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                    wrist = [landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                             landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]

                    angle = self.calculate_angle(shoulder, elbow, wrist)
                    self.count_reps(angle, threshold_down=160, threshold_up=30)

                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'Reps: {}'.format(self.counter),
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    self.draw_landmarks(image, landmarks)

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pose_estimator = PoseEstimator()
    pose_estimator.run()