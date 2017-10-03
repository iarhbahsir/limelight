
import csv
import scipy as sp
import scipy.io
import numpy as np
import pickle

class Preprocessor:

    #initialize with the meta file for the pictures and the directory with the embeddings
    def __init__(self, meta_path, embeddings_path, file_labels_path):
        self.meta_path = meta_path
        self.embeddings_path = embeddings_path
        self.file_labels_path = file_labels_path

    # method to read embeddings and label csv files and create vectors
    def create_base_vectors(self):
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

        # save the vectors as a dict (check if ordered) so they don't need to be processed again
        """vectors = {'image_paths': self.image_paths, 'imdb_ids': self.imdb_ids, 'face_scores': self.face_scores,
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

        cnt = 0;

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

        # if less than 3 images, keep in training
        # 3 images: place one in test
        # more images: place at least one in test and at least one in validation
        # TODO see how many actors have few images and test some manually to check if accuracy is ok

    # method to return tuple of (training, validation, and test data sets)


    # method to return tuple of (embeddings vector, imdb_id vector)

    # method to take in the .pickle files and set up vectors accordingly

# method to take in image location, and align then convert into embeddings to return (static)