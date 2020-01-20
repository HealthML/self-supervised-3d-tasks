from PIL import Image
from pathlib import Path

if __name__ == "__main__":
    basepath = Path("/mnt/mpws2019cl1/kaggle_retina/train")
    (basepath / "resized").mkdir(exist_ok=True, parents=True)
    amount = len(list(basepath.glob("*.jpeg")))
    for index, path in enumerate(basepath.glob("*.jpeg")):
        print(f"{index:06d}/{amount:06d}", end="\r")
        image = Image.open(path)
        image = image.resize((256, 256), resample=Image.LANCZOS)
        image.save(path.parent / "resized" / path.name)
