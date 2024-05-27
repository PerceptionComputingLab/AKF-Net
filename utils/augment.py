import numpy as np
import random
import numbers
import random
import skimage


class Sequential(object):
    """
    Composes several augmentations together.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.

        random_order (bool): Whether to apply the augmentations in random order.
    """

    def __init__(self, transforms, random_order=False):
        self.transforms = transforms
        self.rand = random_order

    def __call__(self, clip):
        if self.rand:
            rand_transforms = self.transforms[:]
            random.shuffle(rand_transforms)
            for t in rand_transforms:
                clip = t(clip)
        else:
            for t in self.transforms:
                clip = t(clip)

        return clip


class Sometimes(object):
    """
    Applies an augmentation with a given probability.

    Args:
        p (float): The probability to apply the augmentation.

        transform (an "Augmentor" object): The augmentation to apply.

    Example: Use this this transform as follows:
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        sometimes(va.HorizontalFlip)
    """

    def __init__(self, p, transform):
        self.transform = transform
        if (p > 1.0) | (p < 0.0):
            raise TypeError('Expected p to be in [0.0 <= 1.0], ' +
                            'but got p = {0}'.format(p))
        else:
            self.p = p

    def __call__(self, clip):
        if random.random() < self.p:
            clip = self.transform(clip)
        return clip


"""
Augmenters that apply video flipping horizontally and
vertically.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.HorizontalFlip(),
                          va.VerticalFlip() ])

List of augmenters:
    * HorizontalFlip
    * VerticalFlip
"""

class HorizontalFlip(object):
    """
    Horizontally flip the video.
    """

    def __call__(self, clip):
        if isinstance(clip["data"], np.ndarray):
            return {"data":np.flip(clip["data"], axis=-1).astype(np.float64), "seg":np.flip(clip["seg"], axis=-1).astype(np.int16)}
        else:
            raise TypeError('Expected numpy.ndarray' +
                            ' but got list of {0}'.format(type(clip["data"])))


class VerticalFlip(object):
    """
    Vertically flip the video.
    """

    def __call__(self, clip):
        if isinstance(clip["data"], np.ndarray):
            return {"data":np.flip(clip["data"], axis=-2).astype(np.float64), "seg":np.flip(clip["seg"], axis=-2).astype(np.int16)}
        else:
            raise TypeError('Expected numpy.ndarray' +
                            ' but got list of {0}'.format(type(clip["data"])))


"""
Augmenters that apply affine transformations.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * RandomRotate
    * RandomResize
    * RandomTranslate
    * RandomShear
"""
class RandomRotate(object):
    """
    Rotate video randomly by a random angle within given boundsi.

    Args:
        degrees (sequence or int): Range of degrees to randomly
        select from. If degrees is a number instead of sequence
        like (min, max), the range of degrees, will be
        (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip["data"], np.ndarray):
            img = clip["data"]
            img_shape = img.shape
            img = img.reshape([-1, img_shape[-2], img_shape[-1]])
            aug_img = []
            for itm in img:
                aug_itm = skimage.transform.rotate(itm, angle)
                aug_img.append(aug_itm)
            aug_img = np.stack(aug_img, axis=0).reshape(img_shape)

            label = clip["seg"].squeeze(axis=0)
            class_ceil = np.max(label)
            class_floor = np.min(label)
            aug_label = []
            for itm in label:
                aug_itm = skimage.transform.rotate(itm, angle, preserve_range=True).astype(np.int16)
                assert np.max(aug_itm) <= class_ceil
                assert np.min(aug_itm) >= class_floor
                aug_label.append(aug_itm)
            aug_label = np.stack(aug_label, axis=0)
            aug_label = aug_label[np.newaxis,:]

            rotated = {"data": aug_img.astype(np.float64), "seg": aug_label.astype(np.int16)}
        else:
            raise TypeError('Expected numpy.ndarray' +
                            'but got list of {0}'.format(type(clip["data"])))

        return rotated

