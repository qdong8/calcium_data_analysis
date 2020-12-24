import numpy as np
import peakutils as pu
import sys
import pickle
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from signal_filters import butter, moving_average


class CaSignal:
    def __init__(self, data, sig_wid=50, positive=True, is_smooth=False, smooth_filter=1, smooth_pm=3, threshold=3.0,
                 index=None, group: dict = None, treatment: OrderedDict = None):
        """
        :param data: 1-D array like
        :param sig_wid:
        :param is_smooth:
        :param smooth_filter:
        :param smooth_pm:
        :param threshold: to filter noise and get the events (times of std)
        :param index: linked to the index of an ROI
        :param group: e.g.  {'label':  'GFP+'; 'parts': 'soma'}
        :param treatment: e.g. {'agonist': [50, 100], 'washout': [100,]}
        """
        self.__smooth = None
        self.smooth_filter = smooth_filter
        self.smooth_pm = smooth_pm
        self.index = index
        self.group = group
        self.sig_width = sig_wid

        b = np.min(data) if np.min(data) > 0 else 1
        self.data = np.array(data / b)
        self.__threshold = threshold
        self.__delta = None  # to filter the events

        self.__baseline = None

        if not positive:
            self.data = - self.data

        if is_smooth:
            self.data = self.smoothed_data

        if treatment is not None:
            self.treatment = treatment
            # self.set_treatment(treatment)
        else:
            self.treatment = OrderedDict()
            self.treatment['untreated'] = [0, len(self.data)]

        # self.__events_total = None
        self.__min_list = OrderedDict()
        self.__events = OrderedDict()
        self.__num_events = OrderedDict()
        self.__num_total = 0
        self.__fre = OrderedDict()
        self.__fwhm = OrderedDict()
        self.__hmxy = OrderedDict()

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        if isinstance(item, str):
            if item in self.treatment:
                return self.data[slice(*self.treatment[item])]
            else:
                raise IndexError

    def __init(self):
        self.__num_total = 0
        self.__events.clear()
        self.__num_events.clear()
        self.__fre.clear()
        self.__fwhm.clear()
        self.__hmxy.clear()

    @property
    def fwhm(self):
        if not self.__fwhm:
            self.__get_fwhm()
        return self.__fwhm

    @property
    def hmxy(self):
        if not self.__hmxy:
            self.__get_fwhm()
        return self.__hmxy

    @property
    def num_events(self):
        if not self.__num_events:
            self.get_peaks()
        return self.__num_events

    @property
    def frequency(self):
        if not self.__fre:
            self.get_peaks()
        return self.__fre

    @property
    def amp(self):
        return self.events['untreated'][1].mean()

    @property
    def frequency_average(self):
        if len(self.treatment) == 1:
            return self.frequency['untreated']
        return self.__num_total/len(self.data)

    @property
    def events(self):
        if not self.__events:
            self.get_peaks()
        return self.__events

    @property
    def delta(self):
        if self.__delta is None:
            self.__delta = max(self.__get_std() * self.__threshold, 0.25)
        return self.__delta

    @delta.setter
    def delta(self, value):
        self.__delta = value
        if self.events is not None:
            self.get_peaks()

    def __get_std(self):
        baseline = self.data - gaussian_filter(self.data, sigma=3)
        return baseline.std()

    @property
    def smoothed_data(self):
        if self.__smooth is not None:
            return self.__smooth
        self.__smooth = self.__filter(self.data, self.smooth_filter, self.smooth_pm)
        return self.__smooth

    @staticmethod
    def __filter(data=None, smooth_filter=0, smooth_pm=3):
        if smooth_filter == 0:
            return butter(data, cutoff=smooth_pm)
        if smooth_filter == 1:
            return moving_average(data, size=smooth_pm)
        if smooth_filter == 2:
            return medfilt(data, int(smooth_pm))
        else:
            return gaussian_filter(data, sigma=smooth_pm)

    def set_treatment(self, treatment):
        start = len(self.data)
        for t in treatment:
            duration = treatment[t]
            start = min(duration[0], start)
            if len(duration) == 1:
                duration.append(len(self.data) - 1)
            self.treatment[t] = duration
        self.treatment['untreated'] = [0, start]

        if not self.__events:
            self.get_peaks()

    def cross_correlation(self, other):
        a = (self.data - np.mean(self.data)) / (np.std(self.data) * len(self.data))
        v = (other.data - np.mean(other.data)) / np.std(other.data)
        return np.correlate(a, v, 'full')

    @staticmethod
    def __peak_det(v, d, x=None):
        """

        :param v:
        :param d:
        :param x:
        :return: (events amplitudes (x, y), local minimum (x, y))
        """
        maxtab = []
        mintab = []

        if x is None:
            x = np.arange(len(v))

        v = np.asarray(v)

        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')

        if not np.isscalar(d):
            sys.exit('Input argument delta must be a scalar')

        if d <= 0:
            sys.exit('Input argument delta must be positive')

        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN

        lookformax = True

        for i in np.arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]

            if lookformax:
                if i == len(v) - 1 and mx > mn + d:
                    maxtab.append((mxpos, mx))
                    continue
                if this < mx - d:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + d:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True

        return np.array(maxtab), np.array(mintab)

    def __get_fwhm(self):
        """

        :return: fwhm[]
        """
        for treatment in self.treatment:
            self.__fwhm[treatment] = []
            self.__hmxy[treatment] = []
            if self.num_events[treatment] == 0:
                continue

            for i in range(min(self.num_events[treatment], len(self.__min_list[treatment][0]))):
                y_peak = self.events[treatment][1][i]
                x_peak = self.events[treatment][0][i]
                if x_peak < 2 or x_peak + 4 > len(self.data):
                    continue
                hm = (y_peak - 1.0) * 0.5 + 1.0     # 0.2 hm
                if i == 0:
                    start = int(np.maximum(0, x_peak - 5))
                else:
                    start = int(np.maximum(self.__min_list[treatment][0][i - 1], x_peak - 5))
                if i == self.num_events[treatment] - 1:
                    end = len(self.data) - 1
                else:
                    end = int(self.__min_list[treatment][0][i])
                x_rise = np.arange(start, x_peak + 1, dtype=int)
                rise = self.data[x_rise]
                x_decay = np.arange(x_peak, end, dtype=int)
                decay = self.data[x_decay]
                rise1 = rise[np.where(rise < hm)]
                x12 = len(rise1) + start
                if x12 == start:
                    continue
                    # xx, yy, x_hm1 = self.spike_rise_fit(x_rise, rise, hm)
                    # self.fit.append((xx, yy))
                else:
                    x11 = x12 - 1
                    y11 = rise[x11 - start]
                    y12 = rise[x12 - start]
                    x_hm1 = x11 + (hm - y11) / (y12 - y11)
                decay1 = decay[np.where(decay > hm)]
                x22 = len(decay1) + x_peak
                if x22 == len(decay) + x_peak:
                    continue
                    # xx, yy, x_hm2 = self.spike_decay_fit(x_decay, decay, hm)
                    # self.fit.append((xx, yy))
                else:
                    x21 = x22 - 1
                    y21 = decay[int(x21 - x_peak)]
                    y22 = decay[int(x22 - x_peak)]
                    if y22 != y21:
                        x_hm2 = x21 + (hm - y21) / (y22 - y21)
                    else:
                        x_hm2 = x21
                    self.__fwhm[treatment].append((x_hm2 - x_hm1))
                    self.__hmxy[treatment].append((x_hm1, x_hm2, hm))

    def get_peaks(self):
        self.__init()
        for treatment in self.treatment:
            peak_xy, min_xy = self.__peak_det(self[treatment], self.delta)
            self.__num_events[treatment] = len(peak_xy)
            self.__num_total += len(peak_xy)
            self.__fre[treatment] = self.__num_events[treatment] / len(self[treatment])
            events = peak_xy.T
            if len(events) > 0:
                events[0] += self.treatment[treatment][0]   # adjust the time
            self.__events[treatment] = events
            self.__min_list[treatment] = min_xy.T

    @property
    def baseline(self):
        if self.__baseline is None:
            self.__baseline = pu.baseline(self.data)
        return self.__baseline

    @baseline.setter
    def baseline(self, value):
        self.__baseline = value

    def adjust_baseline(self):
        data_ = self.data - self.baseline + 1
        self.data = data_/np.min(data_)
        self.__baseline = np.ones_like(self.data)
        if not self.__events:
            self.get_peaks()

    def save(self, path):
        pickle.dump(self, path)


