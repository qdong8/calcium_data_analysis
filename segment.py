from scipy import ndimage
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def detect_peaks(image: np.ndarray):
    """

    :param image: A 2D array
    :return: A boolean mask
    """
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    
    # create the mask of the background
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks


def surroundings(index_list, shape):
    """

    :param index_list: a list of xy pairs eg. [(4,7),(4,8),(4,9),(5,6),(5,7),(5,8),(6,7),(7,7)]
    :param shape: (height, width)
    :return: the pixels outside the input pxs
    """
    l = []
    xlim = shape[0]
    ylim = shape[1]
    for px in index_list:
        x = px[0]
        y = px[1]
        left = (x - 1, y) if x > 1 else px
        right = (x + 1, y) if x < xlim - 2 else px
        up = (x, y + 1) if y < ylim - 2 else px
        down = (x, y - 1) if y > 1 else px
        if left not in index_list:
            l.append(left)
        if right not in index_list:
            l.append(right)
        if up not in index_list:
            l.append(up)
        if down not in index_list:
            l.append(down)
    return l


class Segment:
    def __init__(self, image, c=0, z=0, callback=None, threshold_cc=0.8, threshold_area=6, smooth_size=3,
                 threshold_mxpx=0):
        
        self.image = image

        if self.image.data.shape[0] >= 600: 
            self.data = np.zeros((300, *self.image.data.shape[1:]))
            for i in range(300):
                scale = int(self.image.data.shape[0] / 300)
                self.data[i] = self.image.data[i * scale]
        else:
            self.data = self.image.data
        self.callback = callback
        self.c = c
        self.z = z
        self.size = 0
        self.__smooth_size = int(smooth_size)
        self.frame_shape = self.data[0][z][c].shape

        self.arr = np.transpose(self.data, (1, 2, 3, 4, 0))[self.z][self.c]
        self.arr[np.where(self.arr < self.arr.mean())] = self.arr.mean()
        self.__mxpx = None

        self.__segs = None
        self.__threshold_cc = threshold_cc
        self.__threshold_area = threshold_area
        self.__threshold_mxpx = threshold_mxpx

        self.max_im = np.max(self.arr, axis=2)
        try:
            mean_array = np.mean(self.arr, axis=2) + self.max_im
        except MemoryError:
            print('memory error in segmentation')
            mean_array = self.data[0][self.z][self.c]+self.data[-1][self.z][self.c]  + self.max_im
        if self.__smooth_size > 0:
            self.smoothed_data = ndimage.filters.gaussian_filter(mean_array, sigma=self.__smooth_size, mode='nearest')
        else:
            self.smoothed_data = mean_array

    def set_threshold_mxpx(self, n):
        self.__threshold_mxpx = n
        self.__mxpx = list(self.__max_px())

    @property
    def mxpx(self):
        if self.__mxpx is None:
            self.__mxpx = list(self.__max_px())
        return self.__mxpx

    @property
    def segs(self):
        if self.__segs is None:
            self.__seg()
        return self.__segs

    def __max_px(self):
        """
        :return: [(x1 y1),(x2 y2),(x3 y3)]
        """

        px = np.where(detect_peaks(self.smoothed_data))
        list_px = np.array(px).T

        for i_ in list_px:
            if self.smoothed_data[i_[0], i_[1]] < self.__threshold_mxpx:
                continue
            yield i_[0], i_[1]

    def __is_duplicate(self, px_):
        for i_ in self.segs:
            try:
                if px_ in i_:
                    return True
            except ValueError:
                continue
        return False

    def __seg_px(self, px):
        """
        
        :param px: [(),(),()]
        :return: 
        """

        bad_pixels = []

        while True:
            sur = surroundings(px, self.frame_shape)
            flag = len(px)
            for ii in sur:
                if ii in bad_pixels or ii in px:
                    continue
                
                r = np.corrcoef(self.arr[px[0]], self.arr[ii])
                if r[0][1] > self.__threshold_cc:
                    px.append(ii)
                else:
                    bad_pixels.append(ii)
            if len(px) == flag or len(px) > 1000:
                return px

    def __seg(self):
        self.__segs = []
        i = 0
        for px in self.mxpx:
            i += 1
            if self.__is_duplicate(px):
                continue
            total = len(self.mxpx)
            self.callback(i, total)
            self.__segs.append(self.__seg_px([px]))
        for seg_ in self.__segs[:]:
            if len(seg_) < self.__threshold_area:
                self.__segs.remove(seg_)

