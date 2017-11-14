from PIL import Image
import pyrenn as prn
import numpy as np
from captcha.image import ImageCaptcha
import itertools
import pickle
import os

INPUT_SIZE = 9600  # Captcha size is 160 * 60
OUTPUT_SIZE = 60  # Number of permutations
RAW_INPUT_CHARS = ['A', 'B', 'C', 'D', 'E']
SIZE = 3

# image = ImageCaptcha()
# image.generate_image("abcd")
# image.write('abcd', 'out2.png')
net = prn.CreateNN([INPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE])


def generate_map_dict(size):
    generator = enumerate(itertools.permutations(RAW_INPUT_CHARS, size))
    map_dict = {}
    for index, perm in generator:
        vector_rep = [0] * 60
        vector_rep[index] = 1
        print(vector_rep)
        print("{} {}".format(index,perm))
        output = "".join(perm)
        map_dict[output] = np.array(vector_rep)
        # image = ImageCaptcha()
        # image.generate_image(output)
        # image.write(output, "./images/{}.png".format(output))

    pickle.dump(map_dict, open("map.pk1", "wb"))


def get_out_dir(output_dir):
    """
    Creates output directory for inverted block files
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def generate_captchas(output_folder, size):
    get_out_dir(output_folder)
    generator = enumerate(itertools.permutations(RAW_INPUT_CHARS, size))
    for index, perm in generator:
        output = "".join(perm)
        image = ImageCaptcha()
        image.generate_image(output)
        image.write(output, "./{}/{}.png".format(output_folder, output))


if __name__ == "__main__":
    generate_map_dict(SIZE)
    generate_captchas("test", SIZE)