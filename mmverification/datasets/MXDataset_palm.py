import os.path as osp
from typing import List, Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import is_abs
from torch.utils.data import Dataset

from ..registry import DATASETS


@DATASETS.register_module()
class MXFaceDataset_palm(Dataset):
    def __init__(self, root_dir, local_rank, im_size=112, align='v6', online_align=False):
        super(MXFaceDataset_palm, self).__init__()
        self.transform = augmentation().get_augmentation()
        self.online_align = online_align
        self.padding_prob = 0.2
        self.jitter_prob = 0.2
        self.im_size = im_size

        self.padding = 10
        self.jitter_radius = 5

        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')

        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        self.image_width = 720
        self.image_height = 1140

        # header, _ = mx.recordio.unpack(s)
        # if header.flag > 0:
        #     self.header0 = (int(header.label[0]), int(header.label[1]))
        #     self.imgidx = np.array(range(1, int(header.label[0])))
        #
        # else:
        self.imgidx = np.array(list(self.imgrec.keys))
        self.align_version = align
        if int(os.environ["RANK"]) == 0:
            print("Align Version:{}".format(align))
        if align == 'v5':
            self.align = self.align_v5
        elif align == 'v3':
            self.align = self.align_v3
        elif align == 'v4':
            self.align = self.align_v4
        elif align == 'v6':
            self.align = self.align_v6f

    def jitter(self, point1, point2, point3):

        point1 = self.jitter_point(point1)
        point2 = self.jitter_point(point2)
        point3 = self.jitter_point(point3)

        return point1, point2, point3

    def align_v6f(
            self,
            points: List,
            image: np.ndarray,
            output_width: int = 112,
            output_height: int = 112,
            is_jitter: bool = False,
            padding: int = 0,
            angle: Union[None, int, float] = None) -> np.ndarray:
        """
        Align using 8 points.
        """
        center_point = (int(points[7][0]), int(points[7][1]))
        index_point, ring_point = points[3], points[5]

        if angle is None:
            if index_point[0] > ring_point[0]:
                dy, dx = index_point[1] - ring_point[1], index_point[0] - ring_point[0]
            else:
                dy, dx = ring_point[1] - index_point[1], ring_point[0] - index_point[0]
            angle = np.arctan2(dy, dx) * 180 / np.pi

        # Determine if an additional 180° rotation is needed
        additional_rotation = 180 if points[0][1] < points[3][1] else 0
        total_angle = angle + additional_rotation

        rotation_matrix = cv2.getRotationMatrix2D(center_point, total_angle, 1)

        points_rotated = [list(rotate_point(point, rotation_matrix)) for point in points]

        hand_side = "right" if points_rotated[5][0] > points_rotated[3][0] else "left"
        bias = 15 if hand_side == "right" else -15

        if hand_side == "right":
            point1 = [points_rotated[2][0], points_rotated[3][1]]
            point2 = [points_rotated[6][0], points_rotated[5][1]]
            point3 = points_rotated[0]
        else:
            point1 = [points_rotated[6][0], points_rotated[5][1]]
            point2 = [points_rotated[2][0], points_rotated[3][1]]
            point3 = points_rotated[0]

        post2 = np.float32([
            [padding, padding],
            [output_width + padding, padding],
            [output_width / 2 + padding + bias, output_height + padding]
        ])

        if is_jitter:
            point1, point2, point3 = self.jitter(point1, point2, point3)
        post1 = np.float32([point1, point2, point3])

        # Calculate Affine Transformation matrix
        affine_transform = cv2.getAffineTransform(post1, post2)

        combined_transform = np.dot(np.vstack([affine_transform, [0, 0, 1]]),
                                    np.vstack([rotation_matrix, [0, 0, 1]]))

        crop_roi = cv2.warpAffine(image, combined_transform[:2], (output_width, output_height))
        return crop_roi

    def jitter_point(self, point):
        x, y = point
        x_out = x + random.randint(-self.jitter_radius, self.jitter_radius)
        x_out = min(max(x_out, 0), self.image_width - 1)
        y_out = y + random.randint(-self.jitter_radius, self.jitter_radius)
        y_out = min(max(y_out, 0), self.image_height - 1)
        return [x_out, y_out]

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)

        img = mx.image.imdecode(img).asnumpy()

        if not math.isnan(header.label[0]) and self.online_align:

            key_points = header.label
            key_points = np.reshape(key_points, (-1, 2))
            #
            # pointA = key_points[3]
            # pointB = key_points[5]
            # pointC = key_points[0]
            #
            # A = pointA
            # B = pointB
            # C = pointC
            #
            # AB = (B[0] - A[0], B[1] - A[1])
            # AC = (C[0] - A[0], C[1] - A[1])
            #
            # # Compute the z-component of the cross product of A->B and A->C
            # cross_product_z = AB[0] * AC[1] - AB[1] * AC[0]
            # if cross_product_z == 0:
            #     return self.__getitem__(random.randint(0, len(self.imgidx) - 1))

            is_jitter = False
            if self.jitter_radius > 0 and self.jitter_prob > random.random():
                # pointA = self.jitter_point(pointA)
                # pointB = self.jitter_point(pointB)
                is_jitter = True

            try:
                if self.padding_prob > random.random() and self.padding > 0:

                    # _, img = self.align( pointA, pointB, pointC, img, padding=self.padding)
                    img = self.align(key_points, img, output_width=self.im_size, output_height=self.im_size,
                                     is_jitter=is_jitter, padding=self.padding)
                else:
                    # _, img = self.align(pointA, pointB, pointC, img, padding=0)
                    img = self.align(key_points, img, output_width=self.im_size, output_height=self.im_size,
                                     is_jitter=is_jitter, padding=0)

                if img is None:
                    return self.__getitem__(random.randint(0, len(self.imgidx) - 1))
            except Exception as e:
                print("Error in padding")
                # return self.__getitem__(random.randint(0, len(self.imgidx) - 1))
                return self.__getitem__(random.randint(0, len(self.imgidx) - 1))

        if img.shape[0] != self.im_size or img.shape[1] != self.im_size:
            img = cv2.resize(img, (self.im_size, self.im_size))

        img = self.transform(image=img)['image']

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
        img = torch.from_numpy(img.astype(np.float32))

        label = header.id
        label = int(label)
        # if not isinstance(label, numbers.Number):
        #     label = label[0]

        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.imgidx)
