# -*- coding: utf-8 -*-

import argparse
import sys
import logging.config
import requests
import os
import posixpath
from urlparse import urlsplit
from urllib import unquote
import errno
import tqdm
from bs4 import BeautifulSoup
import re
import gzip
import shutil

class DatasetDownloader(object):
    def __init__(self, **kwargs):
        default_attr = dict(verbose=0)

        allowed_attr = list(default_attr.keys())
        default_attr.update(kwargs)

        for key in default_attr:
            if key in allowed_attr:
                self.__dict__[key] = kwargs.get(key)

        self.logger = DatasetDownloader.get_logger(level=logging.DEBUG,
                                                   verbose=default_attr.get('verbose'))

    def _url2filename(self, url):
        """
        Return basename corresponding to url.
        Ref: https://gist.github.com/zed/c2168b9c52b032b5fb7d
        """
        urlpath = urlsplit(url).path
        basename = posixpath.basename(unquote(urlpath))
        if (os.path.basename(basename) != basename or
                    unquote(posixpath.basename(urlpath)) != basename):
            raise ValueError  # reject '%2f' or 'dir%5Cbasename.ext' on Windows
        return basename

    def _mkdir(self, dir_name):
        """
        create dir_name if doesn't exist
        """
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def _download_file(self, url, dir_name):
        """
        Downloads file located at url inside dir
        Ref: https://stackoverflow.com/a/16696317/353736
        """
        filename = self._url2filename(url) # filename
        cwd = os.getcwd() # save current working directory

        self._mkdir(dir_name)
        os.chdir(dir_name) # change to new directory

        # download
        r = requests.get(url, stream=True)
        total_size = (int(r.headers.get('content-length', 0))/1024)
        with open(filename, 'wb') as f:
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024), total=total_size, unit='B', unit_scale=True):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

        # cleanup
        r.close()
        os.chdir(cwd)
        return filename

    def download_edgar(self):
        self.logger.critical('Downloading EDGAR')

    def download_svds(self):
        self.logger.critical('Downloading SVDS')

        # Directory & URLs
        dir = './svds'
        base_url = 'https://raw.githubusercontent.com/silicon-valley-data-science/datasets/master/'
        schema_url = base_url + 'access_log_schema'
        logs_url = base_url + 'access.log'

        self._download_file(schema_url, dir)
        self._download_file(logs_url, dir)

    def download_secrepo(self):
        self.logger.critical('Downloading SECREPO')

        # Directory & Log prefix
        dir = './secrepo'
        temp_dir = '/tmp/.secrepo'
        base_url = 'http://www.secrepo.com/self.logs/2016/'
        log_file_prefix = 'access.log'

        # Make directory
        self._mkdir(dir)

        # Download in temp_dir & extract in dir
        r = requests.get(base_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        for url in tqdm.tqdm(soup.findAll('a', href=re.compile('^' + log_file_prefix))):
            self.logger.warn(url['href'])
            filename = self._download_file(base_url + url['href'], temp_dir)
            with gzip.open(os.path.join(temp_dir, filename), 'rb') as f_in,\
                    open(os.path.join(dir, os.path.splitext(filename)[0]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # cleanup
        shutil.rmtree(temp_dir)
        r.close()

    @staticmethod
    def get_logger(level=logging.DEBUG, verbose=0):
        logger = logging.getLogger(__name__)

        # File
        fh = logging.FileHandler('dataset-downloader.log', 'w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        fh.setLevel(level)
        logger.addHandler(fh)

        # Stream
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        sh_lvls = [logging.ERROR, logging.WARNING, logging.INFO]
        sh.setLevel(sh_lvls[verbose])
        logger.addHandler(sh)

        return logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset(s) to download',
                        choices=['all', 'edgar', 'svds', 'secrepo'])
    parser.add_argument('--verbose', '-v', help='Logging verbosity level',
                        type=int, default=0)
    args = parser.parse_args()

    downloader = DatasetDownloader(**vars(args))

    if args.dataset == 'all':
        downloader.download_edgar()
        downloader.download_svds()
        downloader.download_secrepo()
    elif args.dataset == 'edgar':
        downloader.download_edgar()
    elif args.dataset == 'svds':
        downloader.download_svds()
    elif args.dataset == 'secrepo':
        downloader.download_secrepo()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
