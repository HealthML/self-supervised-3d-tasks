from functools import partial

from PIL import Image
from pathlib import Path
import sys

from multiprocessing import Pool


def resize_one(path, size=(384, 384), output_dir="resized", callback=None):
    output_dir = Path(output_dir)
    image = Image.open(path)
    image = image.resize(size, resample=Image.LANCZOS)
    image.save(output_dir / path.name)
    if callback is not None:
        callback()
    print(output_dir / path.name)


def main():
    if len(sys.argv) <= 1:
        raise ValueError("at least give an image size")
    arg1 = sys.argv[1]
    basepath = Path("/mnt/mpws2019cl1/kaggle_retina_2019/images")
    output_dir = basepath / f"resized_{arg1}"
    output_dir.mkdir(exist_ok=True, parents=True)

    f = partial(resize_one, size=(int(arg1), int(arg1)), output_dir=output_dir)
    with Pool(10) as p:
        p.map(f, basepath.glob("*.png"))


if __name__ == "__main__":
    main()
