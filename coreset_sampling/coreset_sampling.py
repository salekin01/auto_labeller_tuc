import os
import shutil

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

class_name='speed_sign'
coreset_sample_path = class_name + '/coreset_sample'
remaining_population_path = class_name + '/remaining_population'
total_population_path = class_name + '/total_population'
sample_size = 40
target_size = (640, 640)
image_path_list = []


def load_images_from_folder(total_population_path):
    image_list = []
    for filename in os.listdir(total_population_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(total_population_path, filename)
            img = Image.open(img_path)
            image_list.append(img)
            image_path_list.append(img_path)

    return image_list


def image_to_1D_array(img):
    img = img.resize(target_size)
    img = img.convert('L')
    return np.array(img).flatten()


def move_files(file_indices):
    for idx, image_path in enumerate(image_path_list):
        if idx in file_indices:
            shutil.copy(image_path, coreset_sample_path)
        else:
            shutil.copy(image_path, remaining_population_path)


def coreset_sampling():
    images = load_images_from_folder(total_population_path)
    image_matrix = np.stack([image_to_1D_array(img) for img in images])

    num_images_to_select = sample_size
    kmeans_model = KMeans(n_clusters=num_images_to_select, init='k-means++')

    # Fit the model to the data and get the cluster centers
    kmeans_model.fit(image_matrix)
    cluster_centers = kmeans_model.cluster_centers_

    # Get the selected images from the original dataset (based on the closest cluster center)
    selected_indices = []
    for center in cluster_centers:
        distances = np.linalg.norm(image_matrix - center, axis=1)
        closest_index = np.argmin(distances)
        selected_indices.append(closest_index)

        if closest_index not in selected_indices:
            selected_indices.append(closest_index)

        if len(selected_indices) == num_images_to_select:
            break

    selected_indices.sort()

    print(selected_indices)
    move_files(selected_indices)


if __name__ == '__main__':
    coreset_sampling()

    print("Coreset sampling is successful.")
