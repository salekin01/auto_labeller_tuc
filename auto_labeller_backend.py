import glob
import os
import shutil

import label_studio_sdk
import numpy as np
import torch
import yaml
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
from easyocr import Reader
from label_studio.core.utils.io import get_data_dir
from label_studio_ml import model
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys, is_skipped


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
model.LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

CONFIG_DATA = './config/data.yaml'
TRAIN_DATA = './data/train/'
VAL_DATA = './data/val/'

INIT_WEIGHTS = './config/checkpoints/starting_weights.pt'
TRAINED_WEIGHTS = './config/checkpoints/trained_weights.pt'
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
REPO = './yolov7'
IMAGE_SIZE = (640, 640)

load_dotenv()
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")


class AutoLabellerModel(LabelStudioMLBase):
    def __init__(self, device=DEVICE, img_size=IMAGE_SIZE, repo=REPO, train_output=None, **kwargs):
        super(AutoLabellerModel, self).__init__(**kwargs)
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')

        self.repo = repo
        self.device = device
        self.image_dir = upload_dir
        self.img_size = img_size
        self.label_map = {}

        if os.path.isfile(TRAINED_WEIGHTS):
            self.weights = TRAINED_WEIGHTS
        else:
            self.weights = INIT_WEIGHTS

        print(f"The model initialised with weights: {self.weights}")

        with open(CONFIG_DATA, 'r') as stream:
            data = yaml.safe_load(stream)
            self.label_config = data['names']

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image'
        )
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)
        self.label_attrs = schema.get('labels_attrs')

        if self.label_attrs:
            for label_name, label_attrs in self.label_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

    def _get_image_url(self, task):
        image_url = task['data']['image']
        return image_url

    def label2idx(self, label):
        # convert label according to the index in data.yaml
        for index, item in enumerate(self.label_config):
            if label in item:
                return index
        return -1

    def move_files(self, files, label_img_data, val_percent=0.1):
        print("move files")
        if len(files) == 0:
            return
        sortedfiles = sorted(files, key=str.lower)

        val_percent = int(len(files) * val_percent)

        # Use last img as val if there are less than 5 imgs
        if len(files) < 5:
            val_file = sortedfiles[-1]
            base_path = os.path.basename(val_file)
            dest = os.path.join(label_img_data, base_path)
            shutil.copyfile(val_file, dest)

        for ix, file in enumerate(sortedfiles):
            if len(files) - ix <= val_percent:
                base_path = os.path.basename(file)
                dest = os.path.join(label_img_data, base_path)
                shutil.move(file, dest)

    def reset_train_dir(self, dir_path):
        print("reset train directory")
        if os.path.isfile(os.path.join(dir_path, "labels.cache")):
            os.remove(os.path.join(dir_path, "labels.cache"))

        for dir in os.listdir(dir_path):
            shutil.rmtree(os.path.join(dir_path, dir))
            os.makedirs(os.path.join(dir_path, dir))

    def download_tasks(self, project_id):
        print("download tasks from label studio")
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()
        return tasks

    def extract_data_from_tasks(self, tasks):
        print("extract data from tasks")
        for task in tasks:

            if is_skipped(task):
                continue

            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            image_name = image_path.split("/")[-1]
            img = Image.open(image_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(TRAIN_DATA + "images/" + image_name)

            for annotation in task['annotations']:
                for bbox in annotation['result']:
                    bb_width = (bbox['value']['width']) / 100
                    bb_height = (bbox['value']['height']) / 100
                    x = (bbox['value']['x'] / 100) + (bb_width / 2)
                    y = (bbox['value']['y'] / 100) + (bb_height / 2)
                    label = bbox['value']['rectanglelabels']
                    label_idx = self.label2idx(label[0])

                    with open(TRAIN_DATA + "labels/" + image_name[:-4] + '.txt', 'a') as f:
                        f.write(f"{label_idx} {x} {y} {bb_width} {bb_height}\n")

    def fit(self, tasks, workdir=None, batch_size=4, num_epochs=50, **kwargs):
        print("Start training")
        print(kwargs)

        for dir_path in [TRAIN_DATA, VAL_DATA]:
            self.reset_train_dir(dir_path)

        # check if training is from web hook
        if kwargs.get('data'):
            project_id = kwargs['data']['project']['id']
            tasks = self.download_tasks(project_id)
            self.extract_data_from_tasks(tasks)
        # ML training without web hook
        else:
            self.extract_data_from_tasks(tasks)

        img_files = glob.glob(os.path.join(TRAIN_DATA + 'images/', "*.jpg"))
        label_files = glob.glob(os.path.join(TRAIN_DATA + 'labels/', "*.txt"))

        self.move_files(img_files, VAL_DATA + 'images/')
        self.move_files(label_files, VAL_DATA + 'labels/')

        if 'event' not in kwargs:
            now = datetime.now()
            current_dateTime = now.strftime("%d-%m-%Y_%H-%M-%S")

            self.clear_gpu_memory()
            os.system(
                f"python {self.repo}/train.py --workers 1 --device {self.device} --batch-size {batch_size} --epochs {num_epochs} \
                --data ./config/data.yaml --img {self.img_size[0]} {self.img_size[1]} \
                --cfg ./config/yolov7_custom.yaml --weights {self.weights} --name {current_dateTime} \
                --hyp ./config/hyp.scratch.p5.yaml  --exist-ok")
            shutil.copyfile(f"./runs/train/{current_dateTime}/weights/best.pt", TRAINED_WEIGHTS)  # copy trained weights to checkpoint folder

            self.weights = TRAINED_WEIGHTS
            print(f"The new weights are: {self.weights}")
            self.clear_gpu_memory()

        print("end training")
        return {
            'model_path': TRAINED_WEIGHTS,
        }

    def predict(self, tasks, **kwargs):
        print("Start prediction")
        results = []
        all_scores = []
        if 'trained_weights.pt' not in self.weights:
            return [{
                'result': results,
                'score': 0.0
            }]

        self.clear_gpu_memory()
        self.model = torch.hub.load(self.repo, 'custom', self.weights, source='local', trust_repo=True)

        langs = np.array(["en"])
        self.reader = Reader(langs)

        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            img = Image.open(image_path)
            img_width, img_height = get_image_size(image_path)

            preds = self.model(img, size=img_width)
            preds_df = preds.pandas().xyxy[0]
            print(preds_df)

            for x_min, y_min, x_max, y_max, confidence, class_, name_ in zip(preds_df['xmin'], preds_df['ymin'],
                                                                             preds_df['xmax'], preds_df['ymax'],
                                                                             preds_df['confidence'], preds_df['class'],
                                                                             preds_df['name']):
                img_array = np.array(img)
                ROI = img_array[int(y_min):int(y_max), int(x_min):int(x_max)]
                labelIdx = self.optical_character_recognition(ROI)
                if labelIdx == -1:
                    labelIdx = class_

                output_label = self.label_map.get(name_, name_)

                if output_label not in self.labels_in_config:
                    print(output_label + ' label not present in project config.')
                    continue

                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img_width,
                    "original_height": img_height,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [self.label_config[labelIdx]],
                        'x': x_min / img_width * 100,
                        'y': y_min / img_height * 100,
                        'width': (x_max - x_min) / img_width * 100,
                        'height': (y_max - y_min) / img_height * 100
                    },
                    'score': confidence
                })
                all_scores.append(confidence)
                print(results)

        avg_score = sum(all_scores) / max(len(all_scores), 1)

        self.clear_gpu_memory()
        print("end prediction")
        return [{
            'result': results,
            'score': avg_score
        }]

    def clear_gpu_memory(self):
        torch.cuda.empty_cache()

    def optical_character_recognition(self, image):
        results = self.reader.readtext(image)
        result = -1

        speed_signs = {"20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130"}
        for (bbox, text, prob) in results:
            print("[INFO] {:.4f}: {}".format(prob, text))
            text_lower = text.lower().replace(" ", "")
            if text_lower == "stop":
                result = 1
                break
            if text_lower in speed_signs:
                result = 0
                break
        return result
