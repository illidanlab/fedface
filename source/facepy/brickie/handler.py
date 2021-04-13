"""HTTP Handler for WebViewer.
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

import posixpath
import cgi
import shutil
import mimetypes
from string import Template

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    import SimpleHTTPServer as httpserver
    from urllib import unquote
    from StringIO import StringIO as fio
else:
    import http.server as httpserver
    from urllib.parse import unquote
    from io import BytesIO as fio



class Handler(httpserver.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        httpserver.SimpleHTTPRequestHandler.__init__(self, *args, **kwargs)

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.
        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)
        
        Overide: instead of cwd, use a fixed path
        """

        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        path = posixpath.normpath(unquote(path))
        words = path.split('/')
        words = list(filter(None, words))

        if len(words) == 0:
            path = '/'
        elif len(words) > 0 and words[0]=='localfile':
            path = '/'
            words[0] = ''
        elif len(words) > 0 and words[0]=='buffer':
            path = '/'
        else:
            # path = os.getcwd()
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')

        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir): continue
            path = os.path.join(path, word)

        return path

    # def do_GET(self):
    #     """Serve a GET request.

    #     Override: Do not send 
    #     """

    #     f = self.send_head()
    #     if f:
    #         self.copyfile(f, self.wfile)
    #         f.close()

    def send_head(self):
        """Common code for GET and HEAD commands.

        This sends the response code and MIME headers.

        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.

        Override: use tempalte to show images in memory
        """
        path = self.translate_path(self.path)
        f = None

        if self.path == '/':
            f = self.main_page(path)
        elif path.startswith("/buffer/") and path.endswith('.png'):
            f = self.buffer_image(path)
        else:
            # Original code, return a file or dir
            if os.path.isdir(path):
                if not self.path.endswith('/'):
                    # redirect browser - doing basically what apache does
                    self.send_response(301)
                    self.send_header("Location", self.path + "/")
                    self.end_headers()
                    return None
                for index in "index.html", "index.htm":
                    index = os.path.join(path, index)
                    if os.path.exists(index):
                        path = index
                        break
                else:
                    return self.list_directory(path)
            ctype = self.guess_type(path)
            try:
                # Always read in binary mode. Opening files in text mode may cause
                # newline translations, making the actual size of the content
                # transmitted *less* than the content-length!
                f = open(path, 'rb')
            except IOError:
                self.send_error(404, "File not found")
                return None
            self.send_response(200)
            self.send_header("Content-type", ctype)
            fs = os.fstat(f.fileno())
            self.send_header("Content-Length", str(fs[6]))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
        return f


    def buffer_image(self, path):
        
        file_id = path.split('/')[2]
        file_id, ext = os.path.splitext(file_id)
        file_id = int(file_id)
        png = self.viewer.buffer[file_id]
        f = fio()
        f.write(png)
        f.seek(0)

        ctype = self.extensions_map[ext]
        length = len(png)
        self.send_response(200)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

    def main_page(self, path):
        # if self.viewer.updating:
        #     html = "<h2>I am updating images, wait and refresh the page!</h2>"
        if len(self.viewer._image_groups) == 0:
            html = "<h2>The dataset is empty!</h2>"
        else:
            # Decide the initial Layout
            image_groups = self.viewer._image_groups
            descriptions = self.viewer._descriptions	
            if len(descriptions) != len(image_groups): 
                descriptions = [''] * len(image_groups)

            mean_size = sum([len(g) for g in image_groups]) / float(len(image_groups))
            mean_size = int(round(mean_size))
            if mean_size <= 1:
                img_col, col = 1, 12
            elif mean_size == 2:
                img_col, col = 2, 6
            elif mean_size == 3:
                img_col, col = 3, 4
            else:
                img_col, col = 4, 4

            # Add Templates
            content = ''
            for i, group in enumerate(image_groups):
                content += '<div class="item">'
                content += '<div class="fh5co-desc"> %s </div>' % descriptions[i]
                for imgpath in group:
                    content += '''
                        <div class="no-animate-box">
                            <a href="%s" class="image-popup fh5co-board-img"><img src="%s" alt="" class="size-1of%s"></a>
                        </div>
                    ''' % (imgpath, imgpath, str(img_col))
                content += '</div>'

            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'template.html'), 'r') as tf:
                t = Template(tf.read())
                html = t.substitute(content=content, col1=str(col), col2=str(col))
        
        # Fill in the content into the template
        f = fio()
        encoding = sys.getfilesystemencoding()
        if PYTHON_VERSION == 3:
            html = html.encode(encoding, 'surrogateescape')
        f.write(html)

        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()

        return f
