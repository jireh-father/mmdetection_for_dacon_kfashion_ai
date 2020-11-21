import argparse
import os
import json
import shutil
import glob


def main(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))

    os.makedirs(args.output_dir, exist_ok=True)

    image_dirs = glob.glob(os.path.join(args.image_dir, "*"))

    for i, image_dir in enumerate(image_dirs):
        image_files = glob.glob(os.path.join(image_dir, "*"))
        for j, image_path in enumerate(image_files):
            print(i, len(image_dirs), j, len(image_files))
            shutil.copy(image_path, args.output_dir)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    main(parser.parse_args())
