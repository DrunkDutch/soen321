from PIL import Image
import pyrenn as prn
import numpy as np
from captcha.image import ImageCaptcha
import itertools
import pickle
import os
from math import factorial


# https://stackoverflow.com/questions/16453188/counting-permuations-in-python
def npermutations(input_list, size):
    num = factorial(len(input_list))
    # mults = Counter(input_list).values()
    den = factorial(len(input_list) - size)
    return num / den


INPUT_SIZE = 9600  # Captcha size is 160 * 60
RAW_INPUT_CHARS = ['A', 'B', 'C', 'D', 'E']
SIZE = 3
OUTPUT_SIZE = int(npermutations(RAW_INPUT_CHARS, SIZE))  # Number of permutations
map_dict = pickle.load(open("map.pk1", "rb"))
# image = ImageCaptcha()
# image.generate_image("abcd")
# image.write('abcd', 'out2.png')
net = prn.CreateNN([INPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE])


def dictionary_iterator(dict):
    for key, value in dict.items():
        yield key, value


def get_captcha_from_number(number):
    temp_map = [0]*60
    temp_map[number-1] = 1
    temp_map = np.array(temp_map)
    gen = iter(dictionary_iterator(map_dict))
    done = False
    while not done:
        try:
            for key, value in gen:
                if np.array_equal(temp_map, value):
                    return key
                else:
                    continue
        except StopIteration:
            done = True


def generate_map_dict(size):
    generator = enumerate(itertools.permutations(RAW_INPUT_CHARS, size))
    map_dict = {}
    for index, perm in generator:
        vector_rep = [0] * 60
        vector_rep[index] = 1
        print(vector_rep)
        print("{} {}".format(index, perm))
        output = "".join(perm)
        map_dict[output] = np.array(vector_rep)

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


def prepare_image(output_folder, id):
    number = get_captcha_from_number(id)
    image_location = "./{}/{}.png".format(output_folder, number)
    in_image = Image.open(image_location)
    in_image = in_image.convert('L')
    image_array = np.asarray(in_image).copy()
    # Convert image to black and white
    image_array[image_array < 200] = 0  # Black
    image_array[image_array >= 201] = 255  # White
    vector = np.empty(0, dtype="uint8")
    for ar in image_array:
        vector = np.concatenate((vector, ar),0)
    return vector


if __name__ == "__main__":
    # generate_map_dict(SIZE)
    # generate_captchas("test", SIZE)
    vec = prepare_image("images", 1)
