import argparse
import os
import json


def save_coco(file_path, anno_dict, image_dict, categories, image_id_list):
    coco = {'categories': categories}
    images = []
    annotations = []
    for image_id in image_id_list:
        images.append(image_dict[image_id])
        if image_id not in anno_dict:
            print("skip! image id  not in anno dict")
            continue
        for anno in anno_dict[image_id]:
            annotations.append(anno)

    coco['annotations'] = annotations
    coco['images'] = images
    json.dump(coco, open(file_path, "w+"))


def main(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))

    os.makedirs(args.output_dir, exist_ok=True)

    print("loading coco dataset")
    coco = json.load(open(args.coco_file))
    print("loaded")

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    image_id_list = []
    image_dict = {}
    for image in images:
        image_id = image['id']
        image_id_list.append(image_id)
        image_dict[image_id] = image

    anno_dict = {}
    for anno in annotations:
        image_id = anno['image_id']
        if image_id not in anno_dict:
            anno_dict[image_id] = []
        anno_dict[image_id].append(anno)

    split_idx = round(len(image_id_list) * args.train_ratio)
    train_image_id_list = image_id_list[:split_idx]
    val_image_id_list = image_id_list[split_idx:]

    save_coco(os.path.join(args.output_dir, "train_total.json"), anno_dict, image_dict, categories, image_id_list)
    save_coco(os.path.join(args.output_dir, "train_split.json"), anno_dict, image_dict, categories, train_image_id_list)
    save_coco(os.path.join(args.output_dir, "val_split.json"), anno_dict, image_dict, categories, val_image_id_list)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--coco_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--train_ratio', type=float, default=0.8)

    main(parser.parse_args())
