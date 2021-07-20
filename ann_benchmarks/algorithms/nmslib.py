from __future__ import absolute_import
import os
import nmslib
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN

class NmslibReuseIndex(BaseANN):
    @staticmethod
    def encode(d):
        return ["%s=%s" % (a, b) for (a, b) in d.items()]

    def __init__(self, metric, method_name, index_param, query_param, sketching=False, sketchSize=None):
        print(index_param, query_param)
        self._metric = metric
        if metric == 'jaccard' and not sketching is False:
            self._nmslib_metric = 'jaccard_sparse_%s' % sketching
        else:            
            self._nmslib_metric = {
                'angular': 'cosinesimil', 'euclidean': 'l2', 'jaccard': 'jaccard_sparse'}[metric]
        self._method_name = method_name
        self._save_index = False
        self._index_param = NmslibReuseIndex.encode(index_param)
        self._sketching = sketching
        self._sketchSize = sketchSize
        if query_param is not False:
            self._query_param = NmslibReuseIndex.encode(query_param)
            self.name = ('Nmslib(method_name={}, index_param={}, '
                         'query_param={})'.format(self._method_name,
                                                  self._index_param,
                                                  self._query_param))
        else:
            self._query_param = None
            self.name = 'Nmslib(method_name=%s, index_param=%s)' % (
                self._method_name, self._index_param)

        self._index_name = os.path.join(INDEX_DIR, "nmslib_%s_%s_%s" % (
            self._method_name, metric, '_'.join(self._index_param)))

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
            os.makedirs(d)

    def fit(self, X):
        if self._method_name == 'vptree':
            # To avoid this issue: terminate called after throwing an instance
            # of 'std::runtime_error'
            # what():  The data size is too small or the bucket size is too
            # big. Select the parameters so that <total # of records> is NOT
            # less than <bucket size> * 1000
            # Aborted (core dumped)
            self._index_param.append('bucketSize=%d' %
                                     min(int(len(X) * 0.0005), 1000))

        if self._metric == "jaccard":
            # Pass sets as strings since couldn't get SPARSE_VECTOR to work so far
            if self._sketching == "goldfinger":
                space_params = {'nb_bits': self._sketchSize}
                dist_uint_type = nmslib.DistType.UINT
            elif self._sketching == "fastsim":
                space_params = {'sketch_size': self._sketchSize}
                dist_uint_type = nmslib.DistType.INT
            else:
                space_params = None
                dist_uint_type = nmslib.DistType.INT
            self._index = nmslib.init(
                space=self._nmslib_metric, method=self._method_name, space_params=space_params, data_type=nmslib.DataType.DENSE_VECTOR, dist_uint_type=dist_uint_type)

        else:
            self._index = nmslib.init(
                space=self._nmslib_metric, method=self._method_name)

        self._index.addDataPointBatch(X)

        if os.path.exists(self._index_name):
            print('Loading index from file')
            self._index.loadIndex(self._index_name)
        else:
            self._index.createIndex(self._index_param)
            if self._save_index:
                self._index.saveIndex(self._index_name)
        if self._query_param is not None:
            self._index.setQueryTimeParams(self._query_param)

    def set_query_arguments(self, ef):
        if self._method_name == 'hnsw' or self._method_name == 'sw-graph':
            self._index.setQueryTimeParams(["efSearch=%s" % (ef)])

    def query(self, v, n):
        ids, distances = self._index.knnQuery(v, n)

        return ids

    def batch_query(self, X, n):
        self.res = self._index.knnQueryBatch(X, n)

    def get_batch_results(self):
        return [x for x, _ in self.res]
