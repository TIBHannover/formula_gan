import os
import re
import h5py

import numpy as np

import logging
import pickle


class FilterCacheWriterHandler:
    def __init__(self, cache_folder, length, names=[], shapes=[], encodings=[]):
        self._names = names
        self._shapes = shapes
        self._encodings = encodings

        self._cache_folder = cache_folder
        # TODO error handling

        self._h5_cache = None
        self._index = None

        self._length = length

    def __enter__(self):
        cache_re = r"cache_(\d+).h5"
        caches_index = [
            int(re.match(cache_re, x).group(1)) for x in os.listdir(self._cache_folder) if re.match(cache_re, x)
        ]
        if len(caches_index) == 0:
            cache_index = 0
        else:
            cache_index = sorted(caches_index)[-1] + 1
        self._h5_cache = h5py.File(os.path.join(self._cache_folder, f"cache_{cache_index}.h5"), "a")
        logging.info(f"Cache: Create new FilterCache cache_{cache_index}.h5")

        for i in range(len(self._names)):
            shape = self._shapes[i]
            if not isinstance(shape, (list, set)):
                shape = [shape]
            self._h5_cache.create_dataset(self._names[i], [self._length] + shape, dtype=self._encodings[i])

        self._index = 0
        return self

    def __exit__(self, type, value, tb):
        self._h5_cache.close()
        self._h5_cache = None

    def write(self, key, names, values):
        self._h5_cache["_id"][self._index] = key
        for i in range(len(names)):
            self._h5_cache[names[i]][self._index] = values[i]
        self._index += 1


class FilterCacheWriter:
    def __init__(self, cache_folder, length, names=[], shapes=[], encodings=[]):
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        self._names = names
        self._shapes = shapes
        self._encodings = encodings

        self._cache_folder = cache_folder
        # TODO error handling
        self._length = length

    def __call__(self):
        return FilterCacheWriterHandler(
            cache_folder=self._cache_folder,
            length=self._length,
            names=self._names,
            shapes=self._shapes,
            encodings=self._encodings,
        )


class FilterCacheReaderHandler:
    def __init__(self, cache_folder, names=[]):

        self._names = names

        self._cache_folder = cache_folder
        self._h5_cache = None

        self._index = None
        self._build_top = 100

    def __enter__(self):
        cache_re = r"cache_(\d+).h5"
        caches_index = [
            int(re.match(cache_re, x).group(1)) for x in os.listdir(self._cache_folder) if re.match(cache_re, x)
        ]
        if len(caches_index) == 0:
            return self

        cache_index = sorted(caches_index)[-1]

        index_path = os.path.join(self._cache_folder, f"cache_{cache_index}_index.pickle")
        top_path = os.path.join(self._cache_folder, f"cache_{cache_index}_top.pickle")
        cache_path = os.path.join(self._cache_folder, f"cache_{cache_index}.h5")

        logging.info(f"Cache: FilterCacheReader: Read cache_{cache_index}.h5")
        self._h5_cache = h5py.File(cache_path, "r", swmr=True)

        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self._index = pickle.load(f)

            # with open(top_path, "rb") as f:
            #     self._top_k = pickle.load(f)

        else:
            self._index = {}

            logging.info(f"Cache: FilterCacheReader: Build index")
            for i in range(self._h5_cache["_id"].shape[0]):
                if len(self._h5_cache["_id"][i].item().decode("utf-8")) == 0:
                    continue
                self._index[self._h5_cache["_id"][i].item().decode("utf-8")] = i
            with open(index_path, "wb") as f:
                pickle.dump(self._index, f)

            # logging.info(f"Cache: FilterCacheReader: Compute top-k")

            # self._top_k = {}
            # for x in range(self._h5_cache["prediction"].shape[1]):
            #     top_k = np.argsort(self._h5_cache["prediction"][:, x])[-self._build_top :]
            #     self._top_k[x] = [self._h5_cache["_id"][y].item().decode("utf-8") for y in top_k[:5].tolist()]

            # with open(top_path, "wb") as f:
            #     pickle.dump(self._top_k, f)
        return self

    def __exit__(self, type, value, tb):

        if self._h5_cache is not None:
            self._h5_cache.close()
        self._h5_cache = None
        self._index = None

    def __iter__(self):
        if self._h5_cache is not None:
            for i in range(self._h5_cache[self._names[0]].shape[0]):
                yield self.read(index=i)

    def __contains__(self, key):
        if self._index is None:
            return False
        return key in self._index

    # def in_top_k(self, prediction_index, id):
    #     return id in self._top_k[prediction_index]

    def read(self, key=None):
        result = {}

        if self._index is None:
            return None
        if key not in self._index:
            return None

        for name in self._names:
            result[name] = self._h5_cache[name][self._index[key]]

        return result


class FilterCacheReader:
    def __init__(self, cache_folder, names=[]):
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        self._names = names

        self._cache_folder = cache_folder

    def __call__(self):

        return FilterCacheReaderHandler(self._cache_folder, self._names)

