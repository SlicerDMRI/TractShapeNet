from __future__ import print_function
import numpy
import torch.utils.data as data
import torch
import random
import h5py


class PointSet_pair(data.Dataset):
    def __init__(self, vec, gt, transform=None, n_points=[1024], val=False):
        self.vec = vec
        self.tpvt = gt
        self.transform = transform
        self.n_points = n_points
        self.val = val
        self.invalid_indices = [] 
        print(f"Dataset initialized with {len(self.vec)} items.")

    def __getitem__(self, index: int):
        # Process the first sample
        feat_1, label_1 = self._process_sample(index)

        # Process the second random sample
        index2 = random.randint(0, len(self.vec) - 1)
        feat_2, label_2 = self._process_sample(index2)

        return feat_1, label_1, feat_2, label_2

    def _process_sample(self, index: int):
        feat_points = self.vec[index]
        tracts = list(feat_points.keys())
        feat = None

        for i, tract in enumerate(tracts):
            feat_tract = feat_points[tract]
            if feat_tract.shape[0] > 0:
                id1 = numpy.random.randint(feat_tract.shape[0], size=2048)
                feat1 = feat_tract[id1]
                if feat is None:
                    feat = feat1
                else:
                    feat = numpy.concatenate((feat, feat1), 0)
            else:
                # Collect the index of invalid data
                self.invalid_indices.append(index)
                print(f"Sample {index} has no valid data in tract '{tract}'.")

        label = self.tpvt[index]
        if isinstance(label, numpy.ndarray):
            label = label.tolist() 

        feat_tensor = torch.tensor(feat.T, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return feat_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.vec)

class PointSet(data.Dataset):
    def __init__(self,vec,gt,transform=None,n_points=[1024],val=False):
        self.vec=vec
        self.tpvt=gt
        self.transform = transform
        self.n_points=n_points
        self.val = val

    def __getitem__(self, index: int):
        #print(index)
        feat_points=self.vec[index]
        tracts=list(feat_points.keys())
        #n_points_sub=int(self.n_points/len(tracts))
        feat=None
        for i,tract in enumerate(tracts):
            feat_tract=feat_points[tract]
            id1 = numpy.random.randint(feat_tract.shape[0], size=2048)
            feat1 = feat_tract[id1]
            if feat is None:
                feat=feat1
            else:
                feat=numpy.concatenate((feat,feat1),0)

        label = numpy.array(self.tpvt[index], dtype=numpy.float32)
        label = torch.tensor(label, dtype=torch.float32)
        feat = torch.tensor(feat.T, dtype=torch.float)
        label = label.clone().detach().float()


        return feat,label

    def __len__(self) -> int:
        return len(self.vec)
        
        
class Fiber_pair_vis(data.Dataset):
    def __init__(self,vec,gt,transform=None,n_points=1024):
        self.vec=vec
        self.tpvt=gt
        self.transform = transform
        self.n_points=n_points

    def __getitem__(self, index: int):
        feat_points=self.vec[index]
        feat=feat_points[self.n_points*index:self.n_points*(index+1),:]
        label = self.tpvt[index]
        feat = torch.tensor(feat.T, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        # if self.transform is not None:
        #     img = self.transform(img)
        return feat,label

    def __len__(self) -> int:
        return len(self.vec)
class Fiber_sub(data.Dataset):
    def __init__(self,vec,gt,transform=None,n_points=[1024]):
        self.vec=vec
        self.tpvt=gt
        self.transform = transform
        self.n_points=n_points

    def __getitem__(self, index: int):

        #print(index)
        sub_id=self.vec[index]
        ff = h5py.File('/subjects/{}.h5'.format(sub_id))
        tracts=list(ff.keys())
        feat=None
        for i,tract in enumerate(tracts):
            feat_tract=ff[tract][:]
            id1 = numpy.random.randint(feat_tract.shape[0], size=20000)
            id = random.sample(list(numpy.unique(id1)), self.n_points[i])
            feat1 = feat_tract[id]
            if feat is None:
                feat=feat1
            else:
                feat=numpy.concatenate((feat,feat1),0)

        label = self.tpvt[index]
        feat = torch.tensor(feat.T, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        # if self.transform is not None:
        #     img = self.transform(img)
        return feat,label

    def __len__(self) -> int:
        return len(self.vec)
