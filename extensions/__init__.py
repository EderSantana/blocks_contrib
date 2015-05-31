import numpy as np
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.evaluators import DatasetEvaluator
import cPickle
import logging
try:
    from twitter import Twitter, OAuth
except:
    pass

logger = logging.getLogger(__name__)


class DataStreamMonitoringAndSaving(SimpleExtension, MonitoringExtension):
    """Monitors values of Theano variables on a data stream. Similar
    to `.DataStreamMonitoring` but saves `what_to_save` every time
    the monitored function gets to its lowest value.
    By default monitoring is done before the first and after every epoch.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable`
        The variables to monitor. The variable names are used as record
        names in the logs.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. Be careful not to update any model parameters
        as this is not intended to alter your model in any meaningfull
        way. A typical use case of this option arises when the theano
        function used for evaluation contains a call to
        :func:`~theano.scan` which might have returned shared
        variable updates.
    data_stream : instance of :class:`.DataStream`
        The data stream to monitor on. A data epoch is requested
        each time monitoring is done.
    what_to_save: usually list of `~tensor.TensorSharedVariable`
    to be saved
    path: str with the path to save `what_to_save`
    """
    PREFIX_SEPARATOR = '_'

    def __init__(self, variables, data_stream, what_to_save,
                 path, updates=None, cost_name='cost', **kwargs):

        # kwargs.setdefault("after_epoch", True)
        # kwargs.setdefault("before_first_epoch", True)
        super(DataStreamMonitoringAndSaving, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(variables, updates)
        self.data_stream = data_stream
        self.path = path
        self.what_to_save = what_to_save
        self.validation_cost = variables[0].name
        self.cost_name = cost_name
        self.prev_best = np.finfo('d').max

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")

        if callback_name == "before_epoch" and \
           self.main_loop.log.status['epochs_done'] == 0:
            self.prev_best = value_dict[self.validation_cost]
        if self.prev_best > value_dict[self.validation_cost]:
            logger.info("Saving best model.")
            cPickle.dump(self.what_to_save, file(self.path, 'w'), -1)
            self.add_records(self.main_loop.log, {'Saved Best': 'True'}.items())
            self.prev_best = value_dict[self.validation_cost]
        elif self.prev_best <= value_dict[self.cost_name]:
            self.add_records(self.main_loop.log, {'Saved Best': 'False'}.items())
        cPickle.dump(self.what_to_save, file('last_'+self.path, 'w'), -1)

# TODO delete this
class ValidateAndSave(SimpleExtension):
    """Validates and saves an external model based on a given function

    Parameters
    ----------

    path: str
        Destination of the pickled file
    what_to_save: list
        What should be saved
    validator: function
        A function that will be tested often, when a minimum value is achieved
        ```what_to_save``` is pickled to ```where_to_save```.
    """
    def __init__(self, path, what_to_save, validator, **kwargs):
        self.path = path
        self.what_to_save = what_to_save
        self.validator = validator
        self.valid_history = []

        super(ValidateAndSave, self).__init__(**kwargs)


    def do(self, callback_name, *args):
        self.valid_history.append(self.validator())
        print 'Im printing'
        if self.valid_history[-1] == np.min(self.valid_history):
            cPickle.dump(self.what_to_save, file(self.path,'w'), -1)
            logger.info('Saved Best model with validation: {0:.2f}' \
                        'at Iteration: {1:d}'.format(
                        self.valid_history[-1],
                        self.main_loop.status.epochs_done
            ))
            cPickle.dump(self.what_to_save, file('self.path','w'), -1)
        else:
            logger.info('Validation: {0:.2f} at Iteratrion: {1:d}'.format(
                        self.valid_history[-1],
                        self.main_loop.status.epochs_done
            ))


class TwitterAnnouncer(SimpleExtension):
    """Tweets you announcing the training is done"""
    def __init__(self, token, token_key, con_secret, con_secret_key, **kwargs):
        self.twitter = Twitter(auth=OAuth(token, token_key,
                               con_secret, con_secret_key))
        kwargs.setdefault("after_training", True)
        super(TwitterAnnouncer, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        if which_callback == "after_training":
            iterdone = self.main_loop.status.iterations_done
            cost = self.main_loop.log.current_row.cost
            time = self.main_loop.log.current_row.total_took
            status = r'Deep learning done! | Iter: {0}' \
                     '| Cost: {1:.2f} | Time: {2:.2f}'
            status = status.format(iterdone, cost, time)

            self.twitter.statuses.update(status=status)
