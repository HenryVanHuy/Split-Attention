"""This script constructs a DGL (Deep Graph Library) graph using the triple file "edge.csv," which
represents dependency syntactic relationships. The `buildHomo` function is used for homograph construction,
while the `buildHete` function is employed for heterograph construction."""

import os
import time
from fairseq.models.transformer import BaseFunc
import sys
import dgl


# Homograph Construction, Graph Construction Method Used by SGSE
class buildHomo(BaseFunc):
    def __init__(self, args):
        super().__init__(args)
        self.__getArgs(args)
        self.__process()

    # User parameter passing, passed in the form of a dictionary.
    def __getArgs(self, args: dict):
        self._raw_dir = args['zipPath']

    # Retrieve the mapping between words and DGL (Deep Graph Library) indices (forw_dict),
    # as well as the mapping between DGL indices and words (back_dict).
    def homoNode(self):
        nodePath = os.path.join(self._raw_dir, 'edge.csv')
        # The use of sets is to ensure the uniqueness of word nodes in the DGL (Deep Graph Library) graph.
        nodes = set()
        with open(nodePath, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = self.lineSplit(line, self.sep)
                if len(line) > 1 and line[0] and line[1]:
                    nodes.update(line)

        # Converting to a list is done to ensure that each generated DGL (Deep Graph Library) graph is the same,
        # thereby ensuring the reproducibility of experiments.
        nodes = list(nodes)
        nodes.sort()
        forw_dict, back_dict = self.buildDict(nodes)
        del nodes
        return forw_dict, back_dict

    # Generate a DGL (Deep Graph Library) graph from the triplets relationship using DGL's API interface.
    def __process(self):
        self.forw_dict['_N'], self.back_dict['_N'] = self.homoNode()
        self.edges = self.tranSubg(edge_dir=self._raw_dir)
        self.graph = dgl.graph(self.edges)
        if not os.path.isdir(self._save_dir):
            os.mkdir(self._save_dir)

    # Graph Saving Functionality
    def saveAll(self):
        assert self.graph
        dgl.save_graphs(self._save_path, [self.graph])
        self._savedict('_N')

    # Retrieving Graphs and Mapping Relationships
    def get_data(self):
        return self.graph, self.forw_dict


# Heterograph Construction, potential for future research utilization.
# The functionality of most functions in this context is similar to that of homograph construction.
class buildHete(BaseFunc):
    def __init__(self, args):
        super().__init__(args)
        self.__getArgs(args)
        self.process()

    def __getArgs(self, args: dict):
        self._raw_dir = args['zipPath']

    def heteNode(self, ntype):
        _path = os.path.join(self._raw_dir, 'node_define/' + ntype)
        _path = self.getFilePath(_path)
        with open(_path, 'r', encoding='utf-8')as rf:
            rdata = rf.readlines()
        rdata = map(lambda x: x.strip('\n'), rdata)
        nodes = set(filter(lambda x: x, rdata))
        forw_dict, back_dict = self.buildDict(nodes)

        return forw_dict, back_dict

    def process(self):
        self.ntypeList = self.getNodeType()
        self.etypeList = self.getEdgeType()
        if not self.ntypeList:
            raise ValueError('No nodeType found.')
        if not self.etypeList:
            raise ValueError('No edgeType found.')
        self.graph_dict = {}

        # Constructing a DGL heterogeneous graph based on the types of edges in the heterogeneous graph.
        for ntype in self.ntypeList:
            self.forw_dict[ntype], self.back_dict[ntype] = self.heteNode(ntype)
        for etype in self.etypeList:
            _path = os.path.join(self._raw_dir, 'relations/' + etype)
            ekey = tuple(etype.split('__'))
            self.graph_dict[ekey] = self.tranSubg(edge_dir=_path, etype=etype)
        self.graph = dgl.heterograph(self.graph_dict)
        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir)

    def saveAll(self):
        for i in self.forw_dict:
            self._savedict(i)
        assert self.graph
        dgl.save_graphs(self._save_path, [self.graph])

    # Retrieve Node Types
    def getNodeType(self):
        nodePath = os.path.join(self._raw_dir, 'node_define')
        nodedir = os.listdir(nodePath)
        nodedir = [
            _ for _ in nodedir if os.path.isdir(
                os.path.join(
                    nodePath, _))]
        return nodedir

    # Retrieve Edge Types
    def getEdgeType(self):
        relationPath = os.path.join(self._raw_dir, 'relations')
        filedir = os.listdir(relationPath)
        filedir = [
            _ for _ in filedir if os.path.isdir(
                os.path.join(
                    relationPath, _))]
        _filter = list(filter(lambda x: len(x.split('__')) == 3, filedir))
        if set(filedir) != set(_filter):
            print('Those invalid edgetypes are ignored', set(filedir) - set(_filter))
        return _filter


if __name__ == '__main__':

    args = sys.argv
    if args['graphType'] == 'homo':
        graph = buildHomo(args)
        graph.saveAll()
    elif args['graphType'] == 'hete':
        graph = buildHete(args)
        graph.saveAll()
    else:
        raise ValueError('Wrong graph type.')
