import cv2
import argparse
from ultralytics import YOLO
import multiprocessing
import time
import math

def separate_batches(data, batch_size):
    """
    Разделяет данные на пакеты заданного размера.
    
    Аргументы:
        data (list): Список элементов для разделения на пакеты.
        batch_size (int): Размер каждого пакета.
        
    Возвращает:
        list: Список, содержащий пакеты данных.
    """
    batches = []
    batch_n = len(data) // batch_size + int(len(data) % batch_size > 0) # Определяем количество пакетов
    for i in range(batch_n):
        start_index = i * batch_size  # Начальный индекс текущего пакета
        end_index = min((i + 1) * batch_size, len(data))  # Конечный индекс текущего пакета
        batches.append(data[start_index:end_index])  # Добавляем текущий пакет в список пакетов
    return batches
        

class YoloInference:
    def __init__(self, input_filename, output_filename, multithread):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.multithread = multithread
        self.cap = cv2.VideoCapture(self.input_filename)
        # Проверка успешного открытия входного видеофайла
        if not self.cap.isOpened():
            raise Exception("Ошибка открытия видео потока или файла")
        # Создание объекта для записи видео в указанный выходной файл
        self.writer = cv2.VideoWriter(f"./{self.output_filename}", cv2.VideoWriter_fourcc(*'MP4V'), 30, (640, 480))
        # Проверка успешного создания объекта для записи видео
        if not self.writer.isOpened():
            raise Exception("Ошибка создания видео файла для записи")

    def inference(self, pool_size=4):
        util_frames = [] # Список для хранения кадров
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Изменение размера кадра до 640x480 и добавление его в список
                util_frames.append(cv2.resize(frame, (640, 480)))
            else:
                break
        batch_size = 50
        print(f"Count of frames: {len(util_frames)} Batch Size: {batch_size} Count of batches: {math.ceil(len(util_frames) / batch_size)}")
        util_frames = separate_batches(util_frames, batch_size)
        if self.multithread:
            # Создание пула процессов с заданным количеством процессов
            with multiprocessing.Pool(pool_size) as p:
                results = p.map(self.thread_safe_predict, util_frames) # Параллельная обработка пакетов кадров
            for i, result in enumerate(results):
                batch_frames = result
                for batch_frame in batch_frames:
                    self.writer.write(batch_frame) # Запись каждого обработанного кадра в выходное видео.
        else:
            # Последовательная обработка пакетов
            for frame in util_frames:
                batch_frames = self.thread_safe_predict(frame)
                for batch_frame in batch_frames:
                    self.writer.write(batch_frame)

    def __del__(self):
        self.cap.release()
        self.writer.release()

    @staticmethod
    def thread_safe_predict(input_images):
        """
        Выполняет инференс на входных изображениях с использованием модели YOLO и возвращает обработанные изображения.
        
        Аргументы:
            input_images (list): Список изображений для инференса.
            
        Возвращает:
            list: Список обработанных изображений.
        """
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
