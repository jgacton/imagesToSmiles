labels = ["this is a string", "another string", "N#N", "287926789"]
characters = set(char for label in labels for char in label)

print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)


max_length = max([len(label) for label in labels])


def pad_label(label):
    padded_label = label + (max_length - len(label))*'_'

    return padded_label


def gen_char_map_dic():
    i = 0
    char_map_dic = {}
    for c in characters:
        char_map_dic[c] = i
        i += 1
    c = '_'
    char_map_dic[c] = i

    return char_map_dic


char_map_dic = gen_char_map_dic()
print(char_map_dic)

for i in range(len(labels)):
    label = labels[i]
    label = pad_label(label)
    label = [char_map_dic[c] for c in label]
    string = ''
    for j in range(max_length):
        string = string + str(label[j])
    print(string)
