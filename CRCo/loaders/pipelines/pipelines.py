#
# ----------------------------------------------
import PIL.Image
from .builder import CLS_PIPELINES
import random
import torchvision.transforms as transforms
from mmcls.datasets.pipelines import Compose
from collections.abc import Sequence
from mmcv.utils import build_from_cfg
import numpy as np
import copy


@CLS_PIPELINES.register_module()
class Apply(object):
    def __init__(self, pipeline, prob=1.0):
        if isinstance(pipeline, dict):
            self.pipeline = build_from_cfg(pipeline, CLS_PIPELINES)
        elif isinstance(pipeline, Sequence):
            self.pipeline = Compose(pipeline)
        else:
            raise RuntimeError('wrong type of pipeline when using Apply')
        self.prob = prob

    def __call__(self, result):
        prob = random.random()
        if prob > 0.5:
            return self.pipeline(result)
        else:
            return result

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@CLS_PIPELINES.register_module()
class Hue(object):
    def __init__(self, factor, prob=0.5):
        self.factor = factor
        self.prob = prob

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            tmp_res = PIL.Image.fromarray(results[key])
            tmp_res = transforms.functional.adjust_hue(tmp_res, self.factor)
            results[key] = np.asarray(tmp_res)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_factor={self.factor}, '
        return repr_str


@CLS_PIPELINES.register_module(force=True)
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness.
            brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast.
            contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation.
            saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation].
    """

    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, results):
        brightness_factor = random.uniform(0, self.brightness)
        contrast_factor = random.uniform(0, self.contrast)
        saturation_factor = random.uniform(0, self.saturation)
        hue = random.uniform(0, self.hue)
        color_jitter_transforms = [
            dict(
                type='Brightness',
                magnitude=brightness_factor,
                prob=1.,
                random_negative_prob=0.5),
            dict(
                type='Contrast',
                magnitude=contrast_factor,
                prob=1.,
                random_negative_prob=0.5),
            dict(
                type='ColorTransform',
                magnitude=saturation_factor,
                prob=1.,
                random_negative_prob=0.5),
            dict(
                type='Hue',
                factor=hue,
                prob=1., )
        ]
        random.shuffle(color_jitter_transforms)
        transform = Compose(color_jitter_transforms)
        return transform(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation})'
        repr_str += f'(hue_factor={self.hue}, '
        return repr_str


@CLS_PIPELINES.register_module()
class RandRangeAug(object):
    def __init__(self,
                 policies,
                 num_policies,
                 magnitude_level,
                 total_level=10):
        assert isinstance(num_policies, int), 'Number of policies must be ' \
                                              f'of int type, got {type(num_policies)} instead.'
        assert isinstance(magnitude_level, (int, float)), \
            'Magnitude level must be of int or float type, ' \
            f'got {type(magnitude_level)} instead.'
        assert isinstance(total_level, (int, float)), 'Total level must be ' \
                                                      f'of int or float type, got {type(total_level)} instead.'
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, dict) and 'type' in policy, \
                'Each policy must be a dict with key "type".'

        assert num_policies > 0, 'num_policies must be greater than 0.'
        assert magnitude_level >= 0, 'magnitude_level must be no less than 0.'
        assert total_level > 0, 'total_level must be greater than 0.'

        self.num_policies = num_policies
        self.magnitude_level = magnitude_level
        self.total_level = total_level
        self.policies = policies

    def __call__(self, results):
        if self.num_policies == 0:
            return results
        sub_policy = random.choices(self.policies, k=self.num_policies)
        randomized_sub_policy = []
        #
        for policy in sub_policy:
            processed_policy = copy.deepcopy(policy)
            magnitude_key = processed_policy.pop('magnitude_key', None)
            if magnitude_key is not None:
                minval, maxval = processed_policy.pop('magnitude_range')
                tmp_magnitude_level = random.randint(1, self.magnitude_level)
                magnitude_value = (float(tmp_magnitude_level) /
                                   self.total_level) * float(maxval -
                                                             minval) + minval
                processed_policy.update({magnitude_key: magnitude_value})
            randomized_sub_policy.append(processed_policy)
        sub_policy = Compose(randomized_sub_policy)
        return sub_policy(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(policies={self.policies}, '
        repr_str += f'num_policies={self.num_policies}, '
        repr_str += f'magnitude_level={self.magnitude_level}, '
        repr_str += f'total_level={self.total_level})'
        return repr_str


@CLS_PIPELINES.register_module()
class Identity(object):
    def __init__(self, ):
        pass

    def __call__(self, result):
        return result

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
