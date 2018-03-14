
# import csv
# import scipy as sp
# import scipy.io
import numpy as np
import pickle

class Preprocessor:

    #initialize with the meta file for the pictures and the directory with the embeddings
    def __init__(self, meta_path, embeddings_path, file_labels_path):
        self.meta_path = meta_path
        self.embeddings_path = embeddings_path
        self.file_labels_path = file_labels_path

    # method to read embeddings and label csv files and create vectors
    """def create_base_vectors(self):
        print "Working on imdb info..."

        meta = sp.io.loadmat(self.meta_path, appendmat=True)
        meta = meta['imdb'][0][0]

        self.image_paths = meta[2][0]
        self.imdb_ids = meta[9][0]

        # used to remove bad pictures (if no face, or strong recognition more than one face)
        # TODO find threshold for the removal of picture due to second face
        self.face_scores = meta[6][0]
        self.second_face_scores = meta[7][0]

        # only used after output received
        # TODO see why the actual celeb name is at index - 1 (is there no index of 0?)
        self.all_celeb_names = meta[8][0]

        # probably not needed
        self.names = meta[4][0][0]

        print "Working on embeddings..."

        self.embeddings = []
        with open(self.embeddings_path) as embeddings_csv:
            embeddings_csv_reader = csv.reader(embeddings_csv)
            for embedding in embeddings_csv_reader:
                self.embeddings.append(embedding)

        print "Working on file_labels..."

        self.file_labels = []
        with open(self.file_labels_path) as file_labels_csv:
            file_labels_csv_reader = csv.reader(file_labels_csv)
            for file_label in file_labels_csv_reader:
                self.file_labels.append(file_label[1])

        print "Saving vectors..."

        # save the vectors as a dict (check if ordered) so they don't need to be processed againvectors = {'image_paths': self.image_paths, 'imdb_ids': self.imdb_ids, 'face_scores': self.face_scores,
                   'second_face_scores': self.second_face_scores, 'all_celeb_names': self.all_celeb_names,
                   'names': self.names, 'embeddings': self.embeddings, 'file_labels': self.file_labels}
        with open('vectors.pickle', 'w+') as vectors_file:
            pickle.dump(vectors, vectors_file, protocol=pickle.HIGHEST_PROTOCOL)"""


    # method to create labels vector from meta file, matched it up to the embeddings vector
    def match_vectors(self):
        # zip the embeddings and file_labels and arrange by file_label
       # csv_vectors = zip(self.file_labels, self.embeddings)
       # csv_vectors = sorted(csv_vectors, key=lambda row: row[0])
       # sorted_file_labels, sorted_embeddings = zip(*csv_vectors)

        # create a new embedding vector, which will contain the embeddings in the same order as the meta info
        self.ordered_embeddings = []

        print "Working on matching..."

        cnt = 0

        # go through the zipped meta vectors and look for the embedding in the zipped csv vectors
        for path_element in self.image_paths:
            # modify the path so that it matches that of the csv

            print str(cnt) + "/" + str(len(self.image_paths))
            cnt += 1

            path = str(path_element)
            mod_path = "./aligned-images/" + path[path.find("/") - 2:path.find("jpg")] + "png"
            try:
                embedding_index = self.file_labels.index(mod_path)
                embedding = str(self.embeddings[embedding_index])
            except ValueError:
                print str(embedding_index)
                embedding = ""

        #    embedding_index = np.searchsorted(sorted_file_labels, mod_path)
        #    embedding = sorted_embeddings[embedding_index]

            # write the embedding new embedding vector
            self.ordered_embeddings.append(embedding)

        print "Saving vectors..."

        # embedding vector and imdb_id vectors saved as .pickle
        data_set = {'embedding': self.ordered_embeddings, 'imdb_id': self.imdb_ids}
        with open('data_set.pickle', 'w+') as data_set_file:
                pickle.dump(data_set, data_set_file, protocol=pickle.HIGHEST_PROTOCOL)


    # method to partition embeddings and imdb_id vector in training, validation, and test sets
        # get number of images present for a given actor
        # if less than 2 images, keep in training
        # 2+ images: place one in test
        # more images: place at least one in test and at least one in validation
        # TODO see how many actors have few images and test some manually to check if accuracy is ok
    def partition_data(self, id_counts, converted_sets):
        training_sets = []
        test_sets = []
        temp_id_counts = id_counts.copy()
        for setNum in xrange(len(converted_sets['imdb_id'])):
            if (temp_id_counts[converted_sets['imdb_id'][setNum]] == id_counts[converted_sets['imdb_id'][setNum]]):
                if (id_counts[converted_sets['imdb_id'][setNum]] == 1):
                    training_sets.append((converted_sets['embedding'][setNum], converted_sets['imdb_id'][setNum]))
                else:
                    test_sets.append((converted_sets['embedding'][setNum], converted_sets['imdb_id'][setNum]))
            else:
                training_sets.append((converted_sets['embedding'][setNum], converted_sets['imdb_id'][setNum]))
            temp_id_counts[converted_sets['imdb_id'][setNum]] = temp_id_counts[converted_sets['imdb_id'][setNum]] - 1
        return {'training':training_sets, 'test': test_sets}


# divides sets into groups and appends group label, creates id table
def divideIntoGroups(data_set_path, group_size, specifier=""):
    print "started"
    with open(data_set_path, 'r') as data_file:
        data_set = pickle.load(data_file)
        curr_group = {'in': [], 'out': [], 'out_categorical': []}
        group_num = 0
        curr_num = 1
        embeddings = data_set[0]
        ids = data_set[1]
        group_id_tables = []
        curr_group_id_table = []

        for num in xrange(len(embeddings)):
            if num == 0:
                curr_group_id_table.append(ids[num])
            if num != 0 and (ids[num] != ids[num - 1]):
                curr_num += 1
                curr_group_id_table.append(ids[num])

            curr_group['in'].append(embeddings[num])
            curr_group['out'].append(ids[num])
            curr_group['out_categorical'].append(curr_num-1)

            if curr_num / group_size > group_num:
                print "finished group number " + str(group_num)
                group_filename = "group-" + str(group_num) + "-" + str(specifier) + "-data.pickle"
                with open(group_filename, 'w+') as group_file:
                    pickle.dump(curr_group, group_file, protocol=pickle.HIGHEST_PROTOCOL)
                group_num += 1
                group_id_tables.append(curr_group_id_table)
                curr_group_id_table = []
                curr_group = {'in': [], 'out': [], 'out_categorical': []}
        with open("group_id_tables_small", 'w+') as group_id_tables_file:
            pickle.dump(group_id_tables, group_id_tables_file, protocol=pickle.HIGHEST_PROTOCOL)


def divide_into_groups_small(data_set_path, group_size, specifier=""):
    print "started"
    with open(data_set_path, 'r') as data_file:
        data_set = pickle.load(data_file)
        curr_group = {'in': [], 'out': [], 'out_categorical': []}
        group_num = 0
        curr_num = 1
        embeddings = data_set[0]
        ids = data_set[1]
        group_id_tables = []
        curr_group_id_table = []
        label = 0

        for num in xrange(len(embeddings)):
            if num == 0:
                curr_group_id_table.append(ids[num])
            if num != 0 and (ids[num] != ids[num - 1]):
                curr_num += 1
                label += 1
                curr_group_id_table.append(ids[num])

            if curr_num / group_size > group_num:
                print "finished group number " + str(group_num)
                group_filename = "limelight_data/data/training/group-" + str(group_num) + "-" + str(specifier) + "-data-small.pickle"
                with open(group_filename, 'w+') as group_file:
                    pickle.dump(curr_group, group_file, protocol=pickle.HIGHEST_PROTOCOL)
                group_num += 1
                group_id_tables.append(curr_group_id_table)
                curr_group_id_table = []
                curr_group = {'in': [], 'out': [], 'out_categorical': []}
                label = 0

            curr_group['in'].append(embeddings[num])
            curr_group['out'].append(ids[num])
            curr_group['out_categorical'].append(label)

        with open("limelight_data/data/miscellaneous/group_id_tables_small", 'w+') as group_id_tables_file:
            pickle.dump(group_id_tables, group_id_tables_file, protocol=pickle.HIGHEST_PROTOCOL)

def invert_id_tables(id_tables_path):
    id_tables = pickle.load(open(id_tables_path))
    inverted_id_tables = np.empty([20285, 2])
    for group in xrange(0, len(id_tables)):
        for one_hot in xrange(0, len(id_tables[group])):
            inverted_id_tables[id_tables[group][one_hot]] = [group, one_hot]

    inverted_file_name = "limelight_data/data/miscellaneous/inverted_group_id_tables_small.pickle"
    with open(inverted_file_name, 'w+') as inverted_file:
        pickle.dump(inverted_id_tables, inverted_file, protocol=pickle.HIGHEST_PROTOCOL)


def divide_into_test_groups(test_data_set_path, group_max, specifier="test"):
    print "started"
    curr_group = {'in': [], 'out': [], 'out_categorical': []}
    GROUP_MAX = group_max
    inverted_file_name = "limelight_data/data/miscellaneous/inverted_group_id_tables_small.pickle"
    inverted_file = open(inverted_file_name)
    inverted_id_tables = pickle.load(inverted_file)


    with open(test_data_set_path, 'r') as test_data_file:
        test_data_set = pickle.load(test_data_file)
        for group_num in xrange(GROUP_MAX):
            for pair in test_data_set:
                embedding = pair[0]
                id = pair[1]
                if inverted_id_tables[id][0] == group_num:
                    curr_group['in'].append(embedding)
                    curr_group['out'].append(id)
                    curr_group['out_categorical'].append(inverted_id_tables[id][1])

            print "finished group number " + str(group_num)
            group_filename = "limelight_data/data/test/group-" + str(group_num) + "-" + str(specifier) + "-data.pickle"
            with open(group_filename, 'w+') as group_file:
                pickle.dump(curr_group, group_file, protocol=pickle.HIGHEST_PROTOCOL)
            curr_group = {'in': [], 'out': [], 'out_categorical': []}

# changes the string embeddings read in from the csv to a matrix of floats (static)
def embedding_to_numbers(toConvert):
    converted = []
    for string in toConvert:
        converted_emb = []
        segmented = str(string).split(",")
        segmented[0] = (segmented[0].strip())[1:]
        segmented[len(segmented) - 1] = (segmented[len(segmented) - 1].strip())[:-1]
        for segment in segmented:
            converted_emb.append(float((segment.strip())[1:-1]))
        converted.append(converted_emb)
    return converted

# method to take in image location, and align then convert into embeddings to return (static)

# scale labels
def scale_labels(group_num):
    group_file_name = "limelight_data/data/training/group-" + str(group_num) + "-training-data.pickle"
    group_file = open(group_file_name, 'r')
    group_data = pickle.load(group_file)
    group_data = [np.asarray(group_data['in']), np.asarray(group_data['out_categorical'])]
    scaled_data = {'in': group_data[0], 'out_categorical': [x / 10000.0 for x in group_data[1]]}
    print scaled_data['in'][:50]
    print scaled_data['out_categorical'][:50]
    scaled_file_name = "limelight_data/data/training/group-" + str(group_num) + "-training-data-scaled.pickle"
    with open(scaled_file_name, 'w+') as scaled_file:
        pickle.dump(scaled_data, scaled_file, protocol=pickle.HIGHEST_PROTOCOL)



