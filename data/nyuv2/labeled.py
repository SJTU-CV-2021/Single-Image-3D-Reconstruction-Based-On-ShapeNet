import h5py
import numpy as np
from PIL import Image

def rotate_image(image):
    return image.rotate(-90, expand=True)

class LabeledDataset:
    """Python interface for the labeled subset of the NYU dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path):
        """Opens the labeled dataset file at the given path."""
        self.file = h5py.File(path, mode='r')
        self.color_maps = self.file['images']
        self.depth_maps = self.file['depths']
        self.label_maps = self.file['labels']
        self.instances_maps = self.file['instances']
        self.names = self.file['names'][0]
        self.names = [''.join(chr(i) for i in self.file[obj][:]) for obj in self.names]
        self.names = ['unlabeled'] + self.names
        # print(self.names)

    def close(self):
        """Closes the HDF5 file from which the dataset is read."""
        self.file.close()

    def __len__(self):
        return len(self.color_maps)

    def _get_bounding_box_(self, instances_map, labels_map):
        boxes = {}
        w, h = instances_map.shape
        for i in range(w):
            for j in range(h):
                id = instances_map[i][j]
                tp = labels_map[i][j]
                key = str((id, tp))
                if key in boxes:
                    if i < boxes[key][0]:
                        boxes[key][0] = i
                    if j < boxes[key][1]:
                        boxes[key][1] = j
                    if i > boxes[key][2]:
                        boxes[key][2] = i
                    if j > boxes[key][3]:
                        boxes[key][3] = j
                else:
                    boxes[key] = np.zeros(5)
                    boxes[key][0] = i
                    boxes[key][1] = j
                    boxes[key][2] = i
                    boxes[key][3] = j
                    boxes[key][4] = labels_map[i][j]
        boxes = np.array(list(boxes.values()), dtype=int)
        return boxes[boxes.T[4] != 0]

    def __getitem__(self, idx):
        color_map = self.color_maps[idx]
        color_map = np.moveaxis(color_map, 0, -1)
        color_image = Image.fromarray(color_map, mode='RGB')
        color_image = rotate_image(color_image)

        depth_map = self.depth_maps[idx]
        depth_image = Image.fromarray(depth_map, mode='F')
        depth_image = rotate_image(depth_image)

        labels_map = self.label_maps[idx]
        labels_image = Image.fromarray(labels_map)
        labels_image = rotate_image(labels_image)

        instances_map = self.instances_maps[idx]
        np.set_printoptions(threshold=np.inf)
        # print(instances_map)
        instances_image = Image.fromarray(instances_map)
        instances_image = rotate_image(instances_image)

        labels_map = self.label_maps[idx]
        instances_map = self.instances_maps[idx]
        val_bbox = self._get_bounding_box_(instances_map, labels_map)
        label_dict = [{"bbox": list(bbox[:4]), "class": self.names[bbox[4]]} for bbox in val_bbox]

        return color_image, depth_image, labels_image, instances_image, label_dict


