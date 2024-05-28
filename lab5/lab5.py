import cv2
import argparse
from ultralytics import YOLO
import multiprocessing
import time
import math

def separate_batches(data, batch_size):
    batches = []
    batch_n = len(data) // batch_size + int(len(data) % batch_size > 0)
    for i in range(batch_n):
        batches.append(data[i*batch_size:(i+1)*batch_size if (i+1)*batch_size < len(data) else len(data)])
    return batches
        

class YoloInference:
    def __init__(self, input_filename, output_filename, multithread):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.multithread = multithread
        self.cap = cv2.VideoCapture(self.input_filename)
        self.writer = cv2.VideoWriter(f"./{self.output_filename}", cv2.VideoWriter_fourcc(*'MP4V'), 30, (640, 480))
        if not self.cap.isOpened():
            raise Exception("Error opening video stream or file")

    def inference(self, pool_size=4):
        util_frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                util_frames.append(cv2.resize(frame, (640, 480)))
            else:
                break
        batch_size = 50
        print(f"Count of frames: {len(util_frames)} Batch Size: {batch_size} Count of batches: {math.ceil(len(util_frames) / batch_size)}")
        util_frames = separate_batches(util_frames, batch_size)
        if self.multithread:
            with multiprocessing.Pool(pool_size) as p:
                results = p.map(self.thread_safe_predict, util_frames)
            for i, result in enumerate(results):
                batch_frames = result
                for batch_frame in batch_frames:
                    self.writer.write(batch_frame)
        else:
            for frame in util_frames:
                batch_frames = self.thread_safe_predict(frame)
                for batch_frame in batch_frames:
                    self.writer.write(batch_frame)

    def __del__(self):
        self.cap.release()
        self.writer.release()

    @staticmethod
    def thread_safe_predict(input_images):
        model = YOLO('yolov8s-pose.pt')
        processed_images = []
        for image in input_images:
            inferenced_image = model.predict(image)
            processed_images.append(inferenced_image[0].plot())
        return processed_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='YoloV8S single/multi-threaded inference')
    parser.add_argument("input_filename")
    parser.add_argument("output_filename")
    parser.add_argument('-mt', '--multithread',
                        action='store_true')
    args = parser.parse_args()
    input_filename = args.input_filename
    output_filename = args.output_filename
    multithread = args.multithread

    start_time = time.time()

    yolo_inference = YoloInference(input_filename, output_filename, multithread)
    yolo_inference.inference(pool_size=8)

    print(f"Inference time: {time.time() - start_time}")