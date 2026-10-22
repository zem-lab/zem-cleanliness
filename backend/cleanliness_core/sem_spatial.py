"""Conservative spatial candidate index; exact acceptance stays with callers."""
import numpy as np


class ContourIndex:
    def __init__(self,cell_size):
        self.cell_size=float(cell_size)
        self.cells={}
        self.overflow=[]
        self.entries=[]

    def _cells(self,contour):
        points=np.asarray(contour,np.float64)
        if points.ndim!=2 or points.shape[1:]!=(2,) or len(points)<3 or not np.isfinite(points).all():
            return None
        if np.max(np.abs(points/self.cell_size)) >= 9e15:
            return None
        x0,y0=np.floor(points.min(axis=0)/self.cell_size).astype(np.int64)
        x1,y1=np.floor(points.max(axis=0)/self.cell_size).astype(np.int64)
        if (int(x1)-int(x0)+1)*(int(y1)-int(y0)+1)>10000: return None
        return [(x,y) for x in range(int(x0),int(x1)+1) for y in range(int(y0),int(y1)+1)]

    def add(self,contour,value):
        index=len(self.entries)
        self.entries.append(value)
        cells=self._cells(contour)
        if cells is None: self.overflow.append(index)
        else:
            for cell in cells: self.cells.setdefault(cell,[]).append(index)

    def candidates(self,contour):
        cells=self._cells(contour)
        if cells is None: return self.entries
        indices=set(self.overflow)
        for cell in cells: indices.update(self.cells.get(cell,()))
        return [self.entries[i] for i in sorted(indices)]
