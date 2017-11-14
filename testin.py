import pickle
import numpy as np

map_dict = pickle.load(open("map.pk1", "rb"))
# for key in map_dict.keys():
#     print(map_dict[key])

# print(map_dict)
# print(len(map_dict))


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


if __name__ == "__main__":
    print(get_captcha_from_number(2))
