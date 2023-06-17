import logging

logger = logging.getLogger('gpn')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

formatter = logging.Formatter(_FORMAT)
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)

logger.addHandler(ch)
