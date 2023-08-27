import os
import sys
import numpy as np
import pickle as pkl

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))


class BaseFunc(object):
    def __init__(self, args):
        self.forw_dict = {}
        self.back_dict = {}
        self._save_dir = './temp'
        self._raw_dir = './'
        self.sep = args.get('sep', ',')

    def __getArgs(self, args: dict):
        pass

    def tranSubg(self, edge_dir, etype=None):
        edge_dir = self.getFilePath(edge_dir)
        edges = self.edgeMappping(edge_dir, etype)
        return edges

    @staticmethod
    def getFilePath(_dir):
        _fnamelist = os.listdir(_dir)
        _fname = list(filter(lambda x: x.endswith('.csv'), _fnamelist))
        if not _fname:
            raise FileNotFoundError('NO csv file found in path: ', _dir)
        return os.path.join(_dir, _fname[0])

    def edgeMappping(self, rpath, etype: str):
        if not etype:
            etype = ['_N', '_N']
        else:
            etype = etype.split('__')[0:3:2]
        edges = []

        def errorcheck(_seg, line, etype):
            if not len(line) > 1 and line[0] and line[1]:
                print(line, ' is invalid, and is ignored.')
                return False
            for idx in range(len(_seg)):
                if not _seg[idx]:
                    print('node', line[idx], 'is not defined in nodeType:', etype[idx])
            if not np.array(_seg).all():
                return False
            return True

        with open(rpath, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = self.lineSplit(line, self.sep)
                if errorcheck([line[i] in self.forw_dict[etype[i]] for i in [0, 1]],
                              line,
                              etype):
                    line = tuple(self.forw_dict[etype[i]][line[i]] for i in [0, 1])
                    edges.append(line)
        return edges

    def _savedict(self, ntype):
        if ntype != '_N':
            _path = os.path.join(self._save_dir, ntype)
            if not os.path.exists(_path):
                os.mkdir(_path)
        else:
            _path = self._save_dir
        b_path = os.path.join(_path, 'searchDict.pkl')
        self.pklsave(b_path, self.back_dict[ntype])

    def _extraDict(self, fpath, _raw_dir=True):
        _extraDict = {}
        if _raw_dir:
            _exPath = os.path.join(self._raw_dir, fpath)
        else:
            _exPath = fpath
        with open(_exPath, 'r')as f:
            for node in f:
                nodeF = node.rstrip('\n').split(self.sep, 1)
                feat = nodeF[1]
                node = nodeF[0]
                _extraDict[node] = feat
        _feat_len = len(feat.split(self.sep))
        return _extraDict, _feat_len

    @staticmethod
    def lineSplit(_line, _sep):
        _line = _line.rstrip().rsplit(_sep, 1)
        return _line

    @staticmethod
    def buildDict(_set: list):
        forw_dict = {}
        back_dict = {}
        for i, j in enumerate(_set):
            forw_dict.update({j: i})
            back_dict.update({i: j})
        return forw_dict, back_dict

    @staticmethod
    def getDict(_path, ntype='_N'):
        if ntype != '_N':
            _ = os.path.join(_path, ntype)
        else:
            _ = _path
        _ = os.path.join(_, 'searchDict.pkl')
        if not os.path.exists(_):
            raise FileNotFoundError(
                'No backward dictionary found. Please check your path.')
        else:
            return BaseFunc.pklload(_)

    def process(self):
        self.graph = None
        pass

    @staticmethod
    def pklsave(path, content):
        with open(path, 'wb')as wf:
            pkl.dump(content, wf)

    @staticmethod
    def pklload(path):
        with open(path, 'rb')as rf:
            return pkl.load(rf)

    @property
    def _save_path(self):
        return os.path.join(self._save_dir, 'dglGraph.bin')


if __name__ == '__main__':
    print(BaseFunc.getDict('D:/test123/homo'))

    import ogb.nodeproppred
