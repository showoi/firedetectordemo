from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection
import os

execution_path = os.getcwd()


def train_detection_model():
    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="fire-dataset")
    trainer.setTrainConfig(object_names_array=["fire"], batch_size=8, num_experiments=100, train_from_pretrained_model="yolov3.pt" )
    trainer.trainModel()


def detect_from_image():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "model.pt"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "fire.png"),
                                                 output_image_path=os.path.join(execution_path, "fire-detected.png"),
                                                 minimum_percentage_probability=40)

    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_from_video():
    detector = CustomVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()

    detected_video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "video1.mp4"), frames_per_second=30, output_file_path=os.path.join(execution_path, "video1-detected"), minimum_percentage_probability=40, log_progress=True )

