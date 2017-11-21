from PIL import Image
import pyrenn as prn
import numpy as np
from captcha.image import ImageCaptcha
import itertools
import pickle
import os
from math import factorial
import sys
from keras.models import Sequential
from keras.layers import Dense


# https://stackoverflow.com/questions/16453188/counting-permuations-in-python
def npermutations(input_list, size):
    num = factorial(len(input_list))
    den = factorial(len(input_list) - size)
    return num / den


INPUT_SIZE = 9600  # Captcha size is 160 * 60
RAW_INPUT_CHARS = ['A', 'B', 'C', 'D', 'E']
SIZE = 3
OUTPUT_SIZE = int(npermutations(RAW_INPUT_CHARS, SIZE))  # Number of permutations
map_dict = pickle.load(open("map.pk1", "rb"))
# net = prn.CreateNN([INPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE])


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
    # Convert image to black and white in a binary manner to minimize calculations
    image_array[image_array < 200] = 1  # Black
    image_array[image_array >= 201] = 0  # White
    vector = np.empty(0, dtype="uint8")
    for ar in image_array:
        vector = np.concatenate((vector, ar), 0)
    return vector, number


if __name__ == "__main__":
    """
    To generate the map between string and vector, uncomment generate_map_dict below (not needed, map is stored as 
        map.pk1 and included in repo
    To generate new captchas uncomment generate_captchas and change the first parameter to the desired output directory
    To load and prepare a captcha for testing or training simply change the parameters in the prepare_image line
        first parameter is source folder, second parameter is 1-indexed of the permutation
    To load the net from an existing csv dump uncomment the prn.loadNN statement to load from desired file
        Default for repo is that the net.csv file does not exist and as such this line should remain commented
        Until training and saving is run atleast once
    To train the network uncomment the prn.train line and the prn.saveNN line to train the network and save it as the 
        csv file specified as parameter of prn.saveNN    
    """
    # generate_map_dict(SIZE)
    # generate_captchas("smaller", SIZE)
    vec, captcha = prepare_image("images", 1)
    vec = vec.transpose(0)
    vec = np.reshape(vec, (1, INPUT_SIZE))
    test_array = map_dict[captcha]
    test_array = np.reshape(test_array, (1,60))
    # net = prn.loadNN("net.csv")
    # map_dict = None
    net = Sequential()
    net.add(Dense(24, input_dim=9600, activation='relu'))
    net.add(Dense(8, activation='relu'))
    net.add(Dense(8, activation='relu'))
    net.add(Dense(60, activation='sigmoid'))
    net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    net.fit(vec, test_array, epochs=1000)

    vec, captcha = prepare_image('test', 1)
    vec = vec.transpose(0)
    vec = np.reshape(vec, (1, INPUT_SIZE))
    test_array = map_dict[captcha]
    test_array = np.reshape(test_array, (1,60))
    scores = net.evaluate(vec, test_array)
    print("\n%s: %.2f%%" % (net.metrics_names[1], scores[1] * 100))

    pred = net.predict(vec, batch_size=1, verbose=1)
    print(pred.flatten())
    print(pred.flatten().argmax())
    # net = prn.train_LM(vec, test_array, net, k_max=1)
    # prn.saveNN(net, "net.csv")
