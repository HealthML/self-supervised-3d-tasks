import glob
from functools import partial
from shutil import copyfile

from PIL import Image
from pathlib import Path
import sys

from multiprocessing import Pool


def resize_one(path, size=(224, 224), output_dir="resized", callback=None):
    output_dir = Path(output_dir)
    image = Image.open(path)
    image = image.resize(size, resample=Image.LANCZOS)
    image.save(output_dir / path.name)
    if callback is not None:
        callback()
    print(output_dir / path.name)


def resize_images():
    if len(sys.argv) <= 1:
        raise ValueError("at least give an image size")
    arg1 = sys.argv[1]
    basepath = Path("/mnt/projects/ukbiobank/derived/imaging/retinal_fundus")
    output_dir = basepath / "images_resized_224"
    output_dir.mkdir(exist_ok=True, parents=True)

    f = partial(resize_one, size=(int(arg1), int(arg1)), output_dir=output_dir)
    with Pool(10) as p:
        p.map(f, basepath.glob("*.png"))


def merge_one(path, size=(224, 224), output_dir="resized", callback=None):
    output_dir = Path(output_dir)
    image = Image.open(path)
    image = image.resize(size, resample=Image.LANCZOS)
    image.save(output_dir / path.name)
    if callback is not None:
        callback()
    print(output_dir / path.name)


def merge_in_dir():
    basepath = Path("/mnt/projects/ukbiobank/derived/imaging/retinal_fundus")
    leftpath = basepath / "left/672_left"
    rightpath = basepath / "right/672_right"
    output_dir = basepath / "images_resized_224"

    left_files = leftpath.glob("*.png")
    right_files = rightpath.glob("*.png")
    left_dict = dict()
    for left in left_files:
        id = left.name.split("_")[0]
        left_dict[id] = left
    count = 0
    for right in right_files:
        id = right.name.split("_")[0]
        if id in left_dict:
            count += 1
            copyfile(str(right.resolve()), str(Path(output_dir / right.name).resolve()))
            left = left_dict[id]
            copyfile(str(left.resolve()), str(Path(output_dir / left.name).resolve()))

        if count % 100 == 0:
            print("Processed " + str(count) + " scans so far.")


if __name__ == "__main__":
    merge_in_dir()
