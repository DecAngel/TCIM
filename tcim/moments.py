import cv2

from .utils import PrintSection


MOMENT_SETS = [
    ['m00'],
    ['m10', 'm01'],
    ['m11'],
    ['m20', 'm02'],
    ['m21', 'm12'],
    ['m30', 'm03'],
    ['mu11'],
    ['mu20', 'mu02'],
    ['mu21', 'mu12'],
    ['mu30', 'mu03'],
    ['nu11'],
    ['nu20', 'nu02'],
    ['nu21', 'nu12'],
    ['nu30', 'nu03'],
    ['hu1'],
    ['hu2'],
    ['hu3'],
    ['hu4'],
    ['hu5'],
    ['hu6'],
    ['hu7'],
]
IDENTICAL_THRESHOLD = 1e-2


def get_moments(img):
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments)
    for i in range(7):
        moments[f'hu{i + 1}'] = hu[i][0]
    return moments


def compare_moments(m1, m2):
    res = {}
    for group in MOMENT_SETS:
        for g in group:
            mf, ms = m1[g], m2[g]
            r = (mf - ms) / mf
            name = f'{g} '.rjust(15)
            res[name] = r
        if len(group) == 2:
            g1, g2 = group
            for mf, ms, name in zip(
                    [m1[g1], m1[g2]],
                    [m2[g2], m2[g1]],
                    [f'{g1} vs {g2} '.rjust(15), f'{g2} vs {g1} '.rjust(15)]
            ):
                r = (mf - ms) / mf
                res[name] = r
    return res


class ShowAndCompareMoments:
    def __init__(self):
        self.prev_img = None
        self.prev_moments = None
        self.prev_tag = None

    def record(self, img, tag: str):
        moments = get_moments(img)
        if self.prev_img is not None:
            with PrintSection(f'Compare {self.prev_tag} and {tag}'):
                res = compare_moments(self.prev_moments, moments)
                for k, v in res.items():
                    if abs(v) < IDENTICAL_THRESHOLD:
                        print(f'{k}:IDENTICAL = {v * 100:.2f}%')
                    elif abs(v - 2) < IDENTICAL_THRESHOLD:
                        print(f'{k}:NEGATIVE  = {v * 100:.2f}%')
                    else:
                        # pass
                        print(f'{k}:MISMATCH  = {v * 100:.2f}%')

        self.prev_img = img
        self.prev_moments = moments
        self.prev_tag = tag

        cv2.imshow(tag, img)
        cv2.waitKey(1)

    def __enter__(self):
        self.__init__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
