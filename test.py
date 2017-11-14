from PIL import Image
import pyrenn as prn
import numpy as np
from captcha.image import ImageCaptcha
import itertools
import pickle

INPUT_SIZE = 9600  # Captcha size is 160 * 60
OUTPUT_SIZE = 60  # Number of permutations

# image = ImageCaptcha()
# image.generate_image("abcd")
# image.write('abcd', 'out2.png')
net = prn.CreateNN([INPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE])
raw = ['A', 'B', 'C', 'D', 'E']

generator = enumerate(itertools.permutations(raw, 3))
# print(len(generator))
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


