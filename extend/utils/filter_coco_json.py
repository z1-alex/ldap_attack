import json
from pycocotools.coco import COCO
import os
import numpy as np

def filter_annotations_by_ids(coco, anno_ids):
    # Filter annotations by given annotation ids
    filtered_annotations = [anno for anno in coco.loadAnns(anno_ids)]
    return filtered_annotations

def filter_images_by_annotations(coco, annotations):
    # Get image ids from filtered annotations
    image_ids = list(set([anno['image_id'] for anno in annotations]))
    # Load images based on image ids
    filtered_images = coco.loadImgs(image_ids)
    return filtered_images

def create_small_coco_json(original_coco, anno_ids, output_path):
    # Filter annotations and images
    filtered_annotations = filter_annotations_by_ids(original_coco, anno_ids)
    filtered_images = filter_images_by_annotations(original_coco, filtered_annotations)

    # Filter categories (for person keypoints, there's usually only one category)
    filtered_categories = original_coco.loadCats(original_coco.getCatIds(catNms=['person']))

    # Create a new dictionary for the small COCO dataset
    small_coco_dict = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }

    # Save the new JSON file
    with open(output_path, 'w') as outfile:
        json.dump(small_coco_dict, outfile, indent=4)

def main():
    # Path to the original person_keypoints_train2017.json
    original_json_path = 'coco/annotations/person_keypoints_train2017.json'
    # Path to save the new small JSON file
    output_json_path = 'small_person_keypoints_train2017.json'
    # List of annotation ids to filter
    save_dir = 'extend/dataset'
    save_name = 'train_full_demand_1_person_frcn.npy'
    save_path = os.path.join(save_dir, save_name)
    front_select_list=np.load(save_path)
    front_select_list=front_select_list.tolist()
    anno_ids = front_select_list

    # Initialize COCO object
    coco = COCO(original_json_path)

    # Create small COCO JSON
    create_small_coco_json(coco, anno_ids, output_json_path)

if __name__ == "__main__":
    main()
