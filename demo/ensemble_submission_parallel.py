import json
import numpy as np
from pycocotools import mask as mutils
from pycocotools import coco
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import os
import cv2
from sklearn.metrics import jaccard_score
from multiprocessing import Pool
import pickle

parser = ArgumentParser()
parser.add_argument('--result_files', help='Image file')
parser.add_argument('--output_dir')
parser.add_argument('--thr', type=float, default=0.0)
parser.add_argument('--iou_thr', type=float, default=0.5)
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--use_merge_overlap', action='store_true', default=False)
parser.add_argument('--use_merge', action='store_true', default=False)
parser.add_argument(
    '--device', default='cuda:0', help='Device used for inference')
parser.add_argument(
    '--score-thr', type=float, default=0.3, help='bbox score threshold')
args = parser.parse_args()


def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# Used only for testing.
# This is copied from https://www.kaggle.com/paulorzp/run-length-encode-and-decode.
# Thanks to Paulo Pinto.
def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def annToRLE(segm, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mutils.frPyObjects(segm, h, w)
        rle = mutils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = mutils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def annToMask(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, h, w)
    m = mutils.decode(rle)
    return m


def mask_to_poly(mask):
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # before opencv 3.2
    # contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
    #                                                    cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
    if len(segmentation) == 0:
        return None
    return segmentation


def ensemble(split_data):
    total_data = dict(split_data[0])
    process_id = split_data[1]
    print(total_data)
    print('process_id', process_id)
    print('split data cnt', len(total_data))
    encoded_pixels = []
    img_ids = []
    height = []
    width = []
    category_ids = []
    thr = 800 * 800 * args.thr
    skip_cnt = 0

    for data_idx, image_id in enumerate(total_data):
        if data_idx % 100 == 0:
            print("process id {}. step {}/{}".format(process_id, data_idx, len(total_data)))
        if len(total_data[image_id]) == 1:
            # print("only one instance in a image")
            seg_result = total_data[image_id][0]
            encoded_pixels.append(rle_to_string(rle_encode(mutils.decode(seg_result['segmentation']))))
            img_ids.append(image_id)
            category_ids.append(seg_result['category_id'])
            height.append(seg_result['segmentation']['size'][0])
            width.append(seg_result['segmentation']['size'][1])
        else:
            tmp_data = total_data[image_id]
            overlap_masks = []
            overlap_mask_ids = set()
            if len(tmp_data) == 2:
                mask1 = mutils.decode(tmp_data[0]['segmentation'])
                mask2 = mutils.decode(tmp_data[1]['segmentation'])
                iou = jaccard_score(mask1.flatten(), mask2.flatten())
                if iou == 1.0:
                    # print("same insatnce")
                    seg_result = tmp_data[0]
                    encoded_pixels.append(rle_to_string(rle_encode(mask1)))
                    img_ids.append(image_id)
                    category_ids.append(seg_result['category_id'])
                    height.append(seg_result['segmentation']['size'][0])
                    width.append(seg_result['segmentation']['size'][1])
                    continue
            mask_cache = {}
            for i in range(len(tmp_data) - 1):
                for j in range(i + 1, len(tmp_data)):
                    # for j in range(len(tmp_data)):
                    if i == j:
                        continue
                    if i in mask_cache:
                        mask1 = mask_cache[i]
                    else:
                        mask1 = mutils.decode(tmp_data[i]['segmentation'])
                        mask_cache[i] = mask1

                    if j in mask_cache:
                        mask2 = mask_cache[j]
                    else:
                        mask2 = mutils.decode(tmp_data[j]['segmentation'])
                        mask_cache[j] = mask2

                    iou = jaccard_score(mask1.flatten(), mask2.flatten())
                    if iou >= args.iou_thr:
                        # print('iou', iou)
                        overlap_mask_ids.add(i)
                        overlap_mask_ids.add(j)
                        overlap_masks.append(
                            [[i, mask1], [j, mask2]])
            single_mask_ids = set(range(len(tmp_data))) - overlap_mask_ids
            used_masks = []
            for mask_item1, mask_item2 in overlap_masks:
                if mask_item1[0] in used_masks or mask_item2[0] in used_masks:
                    continue
                # print("merged")
                if args.use_merge:
                    if args.use_merge_overlap:
                        result_mask = mask_item1[1] * mask_item2[1]
                    else:
                        result_mask = ((mask_item1[1] + mask_item2[1]) / 2).astype(np.uint8)
                    selected_id = mask_item1[0]
                else:
                    if mask_item1[0] < mask_item2[0]:
                        result_mask = mask_item1[1]
                        selected_id = mask_item1[0]
                    else:
                        result_mask = mask_item2[1]
                        selected_id = mask_item2[0]

                used_masks.append(mask_item1[0])
                used_masks.append(mask_item2[0])

                encoded_pixels.append(rle_to_string(rle_encode(result_mask)))
                img_ids.append(image_id)
                category_ids.append(tmp_data[selected_id]['category_id'])
                height.append(tmp_data[selected_id]['segmentation']['size'][0])
                width.append(tmp_data[selected_id]['segmentation']['size'][1])
            for i in single_mask_ids:
                # print("single", i)
                seg_result = tmp_data[i]
                if i in mask_cache:
                    mask = mask_cache[i]
                else:
                    mask = mutils.decode(seg_result['segmentation'])
                encoded_pixels.append(rle_to_string(rle_encode(mask)))
                img_ids.append(image_id)
                category_ids.append(seg_result['category_id'])
                height.append(seg_result['segmentation']['size'][0])
                width.append(seg_result['segmentation']['size'][1])

    data = {'ImageId': img_ids,
            'EncodedPixels': encoded_pixels,
            'Height': height,
            'Width': width,
            'CategoryId': category_ids}

    pickle.dump(data, open(os.path.join(args.output_dir, "{}.pkl".format(process_id)), "wb+"))
    print("process id {} done. data cnt {}".format(process_id, len(data['ImageId'])))


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    result_files = args.result_files.split(",")
    total_data = {}
    for ridx, result_file in enumerate(result_files):
        json_data = json.load(open(result_file))
        for i in tqdm(range(len(json_data))):
            image_id = json_data[i]['image_id']
            if image_id not in total_data:
                total_data[image_id] = []
            json_data[i]['ridx'] = ridx
            total_data[image_id].append(json_data[i])

            # if mutils.decode(json_data[i]['segmentation']).sum() < thr:
            #     skip_cnt += 1
            #     print("skip")
            #     continue

    total_cnt = len(total_data)
    each_cnt = total_cnt // args.n
    split_data = []
    total_data_list = list(total_data.items())
    for i in range(args.n):
        split_data.append(total_data_list[i * each_cnt:(i + 1) * each_cnt])
    split_data[0] += total_data_list[args.n * each_cnt:]
    pool = Pool(args.n)
    pool.map(ensemble, list(zip(split_data, list(range(args.n)))))
    print("all done")
    data = {'ImageId': [],
            'EncodedPixels': [],
            'Height': [],
            'Width': [],
            'CategoryId': []}
    for i in range(args.n):
        tmp_data = pickle.load(open(os.path.join(args.output_dir, "{}.pkl".format(i)), "rb"))
        data['ImageId'] += tmp_data['ImageId']
        data['EncodedPixels'] += tmp_data['EncodedPixels']
        data['Height'] += tmp_data['Height']
        data['Width'] += tmp_data['Width']
        data['CategoryId'] += tmp_data['CategoryId']
    submission = pd.DataFrame(data)
    answer_dummy = submission.sample(10)
    print('total length', len(data['ImageId']))
    print("merge done")
    submission.to_csv(os.path.join(args.output_dir, 'submission.csv'), index=False)
    answer_dummy.to_csv(os.path.join(args.output_dir, 'answer_dummy.csv'), index=False)
    # print("skip cnt", skip_cnt)
    print("wrote")


if __name__ == '__main__':
    main()
