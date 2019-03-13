#DO NOT EDIT THIS CODE
import colorsys
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt 
def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = np.stack([height, width, height, width])
    image_dims = np.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def draw_boxes(image, out_scores, out_boxes, out_classes):
    plt.imshow(image)
    class_names = read_classes("coco_classes.txt")
    color_map = {2: "r", 5: "g"}
    legend_map = {}
    for i in reversed(list(range(len(out_classes)))):
        c = out_classes[i]
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        plt.text(left, top, label, color=color_map[c], fontsize=12)
        print(label, (left, top), (right, bottom))
        x = [left, left, right, right, left]
        y = [top, bottom, bottom, top, top]
        line, = plt.plot(x,y, color_map[c])
        legend_map[predicted_class] = line
    classes = list(legend_map.keys())
    values = [legend_map[k] for k in classes]
    plt.legend(values, classes)