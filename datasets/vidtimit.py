import numpy as np
import os
import cPickle
from collections import OrderedDict

from fuel.utils import do_not_pickle_attributes
from fuel.datasets import IndexableDataset
from fuel import config


@do_not_pickle_attributes('indexables')
class VidtimitMouth(IndexableDataset):
    def __init__(self, **kwargs):
        self.sources = ('audio_features', 'video_features',
                        'phrase_targets', 'speaker_targets')

        super(VidtimitMouth, self).__init__(
            OrderedDict(zip(self.sources,
                            self._load_data())),
            **kwargs)

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provide_sources,
                           self._load_data(
                               self.which_dataset,
                               self.which_set))
                           ]

    def _load_data(self):
        """
        which_dataset : choose between 'short' and 'long'
        """
        # Check which_set
        _data_path = os.path.join(config.data_path, 'vidtimit/mouth_vidtimit.pkl')
        data = cPickle.load(file(_data_path, 'r'))

        audio_features = np.asarray(data['audio'])
        video_features = np.asarray(data['video'])
        phrase_targets = np.asarray(data['phrase-target'])
        speaker_targets = np.asarray(data['speaker-target'])
        return audio_features, video_features, phrase_targets, speaker_targets

    def get_data(self, state=None, request=None):
        batch = super(VidtimitMouth, self).get_data(state, request)
        return batch
