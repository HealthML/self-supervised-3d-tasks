import sys
import json
import atexit
import requests
import html
from io import StringIO
from functools import wraps
import logging


class Tee(object):
    def __init__(self, dest1, dest2):
        self.dest1 = dest1
        self.dest2 = dest2

    def __del__(self):
        pass

    def flush(self):
        self.dest1.flush()
        self.dest2.flush()

    def write(self, message):
        self.dest1.write(message)
        self.dest2.write(message)


def shim_outputs():
    logger = logging.getLogger("tensorflow")
    c_stdout = StringIO()
    c_stderr = StringIO()
    ch = logging.StreamHandler(stream=c_stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    atexit.register(ifttt_notify_me, stringio_out=c_stdout, stringio_err=c_stderr)
    return c_stdout, c_stderr


def ifttt_notify_me(
        trigger="model_training_exit",
        key="caN8yF3PRMSkbeaUIZGS-I",
        stringio_out=None,
        stringio_err=None,
):
    data_in = ""
    if not sys.stdin.isatty():
        data_in = sys.stdin.read()

    data_out = ""
    if stringio_out:
        data_out = stringio_out.getvalue()
        data_out = html.escape(data_out)
        data_out = data_out.replace("\n", "<br>")
        stringio_out.close()
    data_err = ""
    if stringio_err:
        data_err = stringio_err.getvalue()
        data_err = html.escape(data_err)
        data_err = data_err.replace("\n", "<br>")
        stringio_err.close()

    request_dict = {"value1": data_in, "value2": data_out, "value3": data_err}
    # post_data = urllib.parse.urlencode(request_dict)
    # post_data = post_data.encode("ascii")
    r = requests.post(
        f"https://maker.ifttt.com/trigger/{trigger}/with/key/{key}", json=request_dict
    )


if __name__ == "__main__":
    ifttt_notify_me()
