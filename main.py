import random
import functools
import json
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

from tcim.transforms import translation, rotation, flip, scale, noise
from tcim.moments import get_moments, compare_moments, ShowAndCompareMoments
from tcim.utils import PrintSection


@functools.lru_cache(1)
def get_sample_image():
    img = cv2.imread('test_image.webp', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (400, 300))
    # _, img = cv2.threshold(img, 127, 255, type=cv2.THRESH_BINARY)
    return img


def transformation_examples():
    img = get_sample_image()
    action_dict = {
        'original': lambda x: x.copy(),
        'identical': lambda x: x.copy(),
        'translated': functools.partial(translation, x=100, y=200),
        'scaled': functools.partial(scale, scale_x=0.6, scale_y=0.8),
        'rotated': functools.partial(rotation, degree=30),
        'flipped': functools.partial(flip, flip_type='vertical'),
        'noised': functools.partial(noise, intensity=0.1),
    }

    with ShowAndCompareMoments() as s:
        for tag, action in action_dict.items():
            img = action(img)
            s.record(img, tag)

    cv2.waitKey()
    cv2.destroyAllWindows()


random_apply_methods = {
    'identical': lambda img: img.copy(),
    'translation': lambda img: translation(img, random.randrange(20, 100), random.randrange(20, 100)),
    'rotation_90_180_270': lambda img: rotation(img, random.choice([90, 180, 270])),
    'rotation_any': lambda img: rotation(img, random.randrange(0, 360)),
    'flip': lambda img: flip(img, random.choice(['horizontal', 'vertical'])),
    'scale': lambda img: scale(img, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)),
    'gaussian_noise': lambda img: noise(img, random.uniform(0.1, 0.5)),
}


def random_apply(img) -> Tuple[int, np.ndarray]:
    i = random.randrange(0, len(random_apply_methods))
    return i, random_apply_methods[list(random_apply_methods.keys())[i]](img)


def load_dataset():
    dataset_path = Path('dataset.json')
    if dataset_path.exists():
        return json.loads(dataset_path.read_text())
    else:
        with PrintSection('Constructing dataset'):
            raw_img = get_sample_image()
            dataset = {'x': [], 'y': []}
            for _ in range(5000):
                _, img1 = random_apply(raw_img)
                y, img2 = random_apply(img1)
                x = compare_moments(get_moments(img1), get_moments(img2))
                dataset['x'].append(list(x.values()))
                dataset['y'].append(y)
            dataset_path.write_text(json.dumps(dataset))
        return dataset


def ten_fold_validation():
    dataset = load_dataset()
    tree = DecisionTreeClassifier()
    with PrintSection('Ten fold validation'):
        cvs = cross_val_score(tree, dataset['x'], dataset['y'], cv=10)
        print(f'ten fold accuracy scores: {cvs}')
        print(f'ten fold average score: {sum(cvs)/len(cvs)}')


def demonstration():
    dataset = load_dataset()
    tree = DecisionTreeClassifier()
    tree.fit(dataset['x'], dataset['y'])
    plt.figure(figsize=(20, 20))
    plot_tree(tree, max_depth=2)
    plt.show()
    raw_img = get_sample_image()
    transform_types = list(random_apply_methods.keys())
    for i in range(100):
        with PrintSection(f'Demo {i}'):
            _, img1 = random_apply(raw_img)
            y, img2 = random_apply(img1)
            x = compare_moments(get_moments(img1), get_moments(img2))
            y_hat = tree.predict([list(x.values())])[0]
            print(f'Truth: {transform_types[y]}, Predicted: {transform_types[y_hat]}')
            img1 = cv2.putText(
                img1, f'Truth: {transform_types[y]}', (0, 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(255, ), thickness=2
            )
            img2 = cv2.putText(
                img2, f'Predict: {transform_types[y_hat]}', (0, 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(255,), thickness=2
            )
            cv2.imshow('Before', img1)
            cv2.imshow('After', img2)
            key = cv2.waitKey()
            cv2.destroyAllWindows()
            if key == ord('q'):
                break


if __name__ == '__main__':
    transformation_examples()
    ten_fold_validation()
    demonstration()
