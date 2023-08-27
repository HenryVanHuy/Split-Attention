import os
import time
from fairseq.models.transformer import BaseFunc
import sys
import dgl


class buildHomo(BaseFunc):
    def __init__(self, args):
        super().__init__(args)
        self.__getArgs(args)
        self.__process()

    def __getArgs(self, args: dict):
        self._raw_dir = args['zipPath']

    def homoNode(self):
        nodePath = os.path.join(self._raw_dir, 'edge.csv')
        nodes = set()
        with open(nodePath, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = self.lineSplit(line, self.sep)
                if len(line) > 1 and line[0] and line[1]:
                    nodes.update(line)
        nodes = list(nodes)
        nodes.sort()
        forw_dict, back_dict = self.buildDict(nodes)
        del nodes
        return forw_dict, back_dict

    def __process(self):
        initime = time.time()
        self.forw_dict['_N'], self.back_dict['_N'] = self.homoNode()
        end_time = time.time()
        # print('Mapping dict built. time used: ', end_time - initime)
        start_time = end_time
        self.edges = self.tranSubg(edge_dir=self._raw_dir)
        end_time = time.time()
        # print('Edge transformed. time used: ', end_time - start_time)
        start_time = end_time
        self.graph = dgl.graph(self.edges)
        end_time = time.time()
        # print('Graph built. time used: ', end_time - start_time)
        # print('Process finished, total time used: ', end_time - initime)
        if not os.path.isdir(self._save_dir):
            os.mkdir(self._save_dir)

    def saveAll(self):
        assert self.graph
        dgl.save_graphs(self._save_path, [self.graph])
        self._savedict('_N')

    def get_data(self):
        return self.graph, self.forw_dict


class buildHete(BaseFunc):

    def __init__(self, args):
        super().__init__(args)
        self.__getArgs(args)
        self.process()

    def __getArgs(self, args: dict):
        # self._save_dir = args['savePath']
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
        initime = time.time()
        self.ntypeList = self.getNodeType()
        self.etypeList = self.getEdgeType()
        if not self.ntypeList:
            raise ValueError('No nodeType found.')
        if not self.etypeList:
            raise ValueError('No edgeType found.')

        end_time = time.time()
        print(
            'nodeType and edgeType got, ErrorCheck pasted. time_used: ',
            end_time - initime)

        start_time = end_time
        self.graph_dict = {}
        for ntype in self.ntypeList:
            self.forw_dict[ntype], self.back_dict[ntype] = self.heteNode(ntype)
        end_time = time.time()
        print('mapping dict built. time used: ', end_time - start_time)

        start_time = end_time
        for etype in self.etypeList:
            _path = os.path.join(self._raw_dir, 'relations/' + etype)
            ekey = tuple(etype.split('__'))
            self.graph_dict[ekey] = self.tranSubg(edge_dir=_path, etype=etype)
        end_time = time.time()
        print('mapping finished, time used: ', end_time - start_time)
        start_time = end_time
        self.graph = dgl.heterograph(self.graph_dict)
        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir)
        end_time = time.time()
        print('graph built, time used: ', end_time - start_time)
        print('process finished, total time used: ', end_time - initime)

    def saveAll(self):
        for i in self.forw_dict:
            self._savedict(i)
        assert self.graph
        dgl.save_graphs(self._save_path, [self.graph])

    def getNodeType(self):
        nodePath = os.path.join(self._raw_dir, 'node_define')
        nodedir = os.listdir(nodePath)
        nodedir = [
            _ for _ in nodedir if os.path.isdir(
                os.path.join(
                    nodePath, _))]
        return nodedir

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
