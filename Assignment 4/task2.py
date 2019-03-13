import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from task2_tools import read_predicted_boxes, read_ground_truth_boxes

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    x_1 = max(prediction_box[0], gt_box[0])
    y_1 = max(prediction_box[1], gt_box[1])
    x_2 = min(prediction_box[2], gt_box[2])
    y_2 = min(prediction_box[3], gt_box[3])

    area_intersection = max(0, x_2 - x_1) * max(0, y_2 - y_1)
    area_pred = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    return area_intersection / float(area_pred + area_gt - area_intersection)

def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fp == 0:
        return 0
    else:
        return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # match_pred = np.array([])
    # match_gt = np.array([])
    # match_iou = np.array([])
    #
    # for pred_box in prediction_boxes:
    #     for gt_box in gt_boxes:
    #         iou = calculate_iou(pred_box, gt_box)
    #         if iou >= iou_threshold:
    #             np.append(match_pred,pred_box)
    #             np.append(match_gt,gt_box)
    #             np.append(match_iou, iou)
    #
    # idx_iou = match_iou.argsort()
    # match_pred = match_pred[idx_iou[::-1]]
    # match_gt = match_gt[idx_iou[::-1]]
    # match_iou = match_iou[idx_iou[::-1]]
    #
    # return match_pred, match_gt
    match_pred = []
    match_gt = []
    match_iou = []

    for pred_box in prediction_boxes:
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                match_pred.append(pred_box)
                match_gt.append(gt_box)
                match_iou.append(iou)

    idx_iou = np.array(match_iou).argsort()
    match_pred = np.array(match_pred)[idx_iou[::-1]]
    match_gt = np.array(match_gt)[idx_iou[::-1]]

    return match_pred, match_gt


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    pred_match, gt_match = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    num_true_pos = pred_match.shape[0]
    num_false_pos = prediction_boxes.shape[0] - num_true_pos
    num_false_neg = gt_boxes.shape[0] - num_true_pos
    return {"true_pos": num_true_pos, "false_pos": num_false_pos, "false_neg": num_false_neg}


def calculate_precision_recall_all_images(all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images.
       
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Find total true positives, false positives and false negatives
    # over all images

    # Compute precision, recall
    total_true_pos, total_false_pos, total_false_neg = 0, 0, 0
    for image_pred_boxes, image_gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        num_dict = calculate_individual_image_result(image_pred_boxes, image_gt_boxes, iou_threshold)

        total_true_pos += num_dict["true_pos"]
        total_false_pos += num_dict["false_pos"]
        total_true_pos += num_dict["true_pos"]

    precision = calculate_precision(total_true_pos, total_false_pos, total_false_neg)
    recall = calculate_recall(total_true_pos, total_false_pos, total_false_neg)

    return precision, recall

def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the precision-recall curve over all images. Use the given
       confidence thresholds to find the precision-recall curve.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both np.array of floats.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    raise NotImplementedError


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    raise NotImplementedError


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
