"""Server class for visualizing images and datasets.
"""
# MIT License
# 
# Copyright (c) 2018 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import sys
import time
import multiprocessing
import threading
import numpy as np
import scipy.misc

from .handler import Handler

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    import SocketServer as socketserver
    from StringIO import StringIO
else:
    import socketserver
    from io import StringIO
    from io import BytesIO


def create_png(img):
    import zlib, struct

    height, width = tuple(img.shape[0:2])

    alpha = 255 * np.ones([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    img = np.concatenate([img, alpha], axis=2)[::-1,:,:]
    buf = img.tobytes()

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(
        b'\x00' + buf[span:span + width_byte_4]
        for span in range((height - 1) * width_byte_4, -1, - width_byte_4)
    )

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    png = b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

    return png


def make_handler_class_from_args(viewer):
    class BoundHandler(Handler):
        def __init__(self, *args, **kwargs):
            self.viewer = viewer
            Handler.__init__(self, *args, **kwargs)
    return BoundHandler 

class WebViewer:
    def __init__(self, image_groups=None, descriptions=None, port=8000):
        self.port = port
        self.server = None
        self.serve_proc = None
        self.stderr = StringIO()
        self.stdout = StringIO()
        self._image_groups = []
        self._descriptions = []
        self.buffer = None
        self.set_images(image_groups, descriptions)

    def release(self):
        if self.serve_proc is not None:
            self.serve_proc.terminate()
            self.serve_proc = None
            print('server closed')


    def set_images(self, image_groups=None, descriptions=None):

        valid_array = lambda x: type(x)==list or type(x)==np.ndarray
        valid_str = lambda x: type(x)==str or type(x)==np.string_ or type(x)==np.str_

        # Check the input format is valid
        if image_groups is not None:
            assert valid_array(image_groups) and all([valid_array(g) for g in image_groups]), \
                "image_groups must be a list of list!"
        if descriptions is not None:
            assert valid_array(descriptions) and all([valid_str(d) for d in descriptions]), \
                "descriptions must be a list of string!"

        if image_groups is None: image_groups = []
        if descriptions is None: descriptions = []

        self._image_groups = []
        self._descriptions = []

        count = 0
        self.buffer = []
        for g in image_groups:
            group = []
            for img in g:
                assert valid_str(img) or type(img)==np.ndarray, \
                    'Each image should be either a string or np.ndarray!'
                if type(img)==np.ndarray:
                    img = np.array(scipy.misc.toimage(img))
                    self.buffer.append(create_png(img))
                    group.append('/buffer/%d.png' % count)
                    count += 1
                else:
                    group.append('/localfile/' + img)
            self._image_groups.append(group)

        for desc in descriptions:
            self._descriptions.append(desc)

        self.release()

    def show_images(self, image_groups=None, descriptions=None):
        self.set_images(image_groups, descriptions)
        time.sleep(1)
        self.serve_background()

    def serve_background(self):        
        self.release()
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        def serve():
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            server = socketserver.ThreadingTCPServer(('localhost', self.port), make_handler_class_from_args(self))
            print("running server at process %d" % os.getpid())
            server.serve_forever()

        self.serve_proc = multiprocessing.Process(target=serve, args=())
        self.serve_proc.daemon = True
        self.serve_proc.start()
        print("serving at port %d" % self.port)

    def serve_forever(self):
        self.release()
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        server = socketserver.ThreadingTCPServer(('localhost', self.port), make_handler_class_from_args(self))
        print("serving at port %d" % self.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
            print('server closed')




