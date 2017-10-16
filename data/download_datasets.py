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
import zipfile
import concurrent.futures


class DatasetDownloader(object):
    def __init__(self, **kwargs):
        default_attr = dict(verbose=0, destination=os.getcwd())

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
        Downloads file located at url inside dir.
        Directory must exist before calling this function
        Ref: https://stackoverflow.com/a/16696317/353736
        """

        # download
        with requests.get(url, stream=True) as r:
            if r.status_code == requests.codes.ok:
                filename = self._url2filename(url) # filename
                with open(os.path.join(dir_name, filename), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                return filename
            else:
                return None

    def _download_file_with_monitor(self, url, dir_name):
        """
        Downloads file located at url inside dir.
        Directory must exist before calling this function
        Ref: https://stackoverflow.com/a/16696317/353736
        """

        # download
        with requests.get(url, stream=True) as r:
            if r.status_code == requests.codes.ok:
                filename = self._url2filename(url) # filename
                total_size = (int(r.headers.get('content-length', 0))/1024)
                with open(os.path.join(dir_name, filename), 'wb') as f:
                    for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024), total=total_size, unit='B', unit_scale=True):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                return filename
            else:
                return None

    def download_edgar(self):
        self.logger.debug('Downloading EDGAR')

        # Download dir, a subdir with year will be created
        dir = os.path.join(self.destination, 'edgar')
        temp_dir = '/tmp/.edgar'

        # List of all log-files
        list_url = 'https://www.sec.gov/files/EDGAR_LogFileData_thru_Dec2016.html'
        # Base URL of actual dataset
        base_url = 'www.sec.gov/dera/data/Public-EDGAR-log-file-data/'

        year = raw_input('Enter the year [2003 ~ 2016]: ')
        if int(year) in range(2003, 2017):
            # Make directory
            dir = os.path.join(dir, year)
            self._mkdir(dir)
            self._mkdir(temp_dir)

            with requests.get(list_url) as r:
                soup = BeautifulSoup(r.text, 'html.parser')
            urls = [x for x in soup.body.string.split('\r\n') if x.startswith(year, len(base_url))]

            # Schedule download
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
            future_to_filename = {executor.submit(self._download_file, 'http://' + url, temp_dir): url for url in urls}

            # Download in temp_dir & extract in dir
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_filename), total=len(future_to_filename)):
                url = future_to_filename[future]
                try:
                    filename = future.result()
                    if filename is None:
                        raise ValueError
                    csv_filename = os.path.splitext(filename)[0] + '.csv'
                    zip_path = os.path.join(temp_dir, filename)

                    with zipfile.ZipFile(zip_path) as logzip:
                        with logzip.open(csv_filename, 'r') as f_in, \
                                open(os.path.join(dir, csv_filename), 'w') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(zip_path)
                except Exception as exc:
                    self.logger.critical('Unable to download {0}: {1}'.format(url, exc))

            # cleanup
            executor.shutdown(True)
            shutil.rmtree(temp_dir)


    def download_svds(self):
        self.logger.debug('Downloading SVDS')

        # Directory & URLs
        dir = os.path.join(self.destination, 'svds')
        base_url = 'https://raw.githubusercontent.com/silicon-valley-data-science/datasets/master/'
        schema_url = base_url + 'access_log_schema'
        logs_url = base_url + 'access.log'

        # Make directory
        self._mkdir(dir)

        # Download
        try:
            if self._download_file_with_monitor(schema_url, dir) is None:
                raise ValueError
            if self._download_file_with_monitor(logs_url, dir) is None:
                raise ValueError
        except Exception as exc:
            self.logger.critical('Unable to download {0}: {1}'.format(logs_url, exc))


    def download_secrepo(self):
        self.logger.debug('Downloading SECREPO')

        # Directory & Log prefix
        dir = os.path.join(self.destination, 'secrepo')
        temp_dir = '/tmp/.secrepo'
        base_url = 'http://www.secrepo.com/self.logs/2016/'
        log_file_prefix = 'access.log'

        # Make directory
        self._mkdir(dir)
        self._mkdir(temp_dir)

        with requests.get(base_url) as r:
            soup = BeautifulSoup(r.text, 'html.parser')
        urls = soup.findAll('a', href=re.compile('^' + log_file_prefix))

        # Schedule download
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        future_to_filename = {executor.submit(self._download_file, base_url + url['href'],
                                              temp_dir): base_url + url['href'] for url in urls}

        # Download in temp_dir & extract in dir     
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_filename), total=len(future_to_filename)):
            url = future_to_filename[future]
            try:
                filename = future.result()
                if filename is None:
                    raise ValueError
                temp_path = os.path.join(temp_dir, filename)

                with gzip.open(temp_path, 'rb') as f_in, \
                        open(os.path.join(dir, os.path.splitext(filename)[0]), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(temp_path)
            except Exception as exc:
                self.logger.critical('Unable to download {0}: {1}'.format(url, exc))

        # cleanup
        executor.shutdown(True)
        shutil.rmtree(temp_dir)


    def download_almhuette_raith(self):
        self.logger.debug('Downloading Almhuette-Raith')

        # Directory & URLs
        dir = os.path.join(self.destination, 'almhuette_raith')
        logs_url = 'http://www.almhuette-raith.at/apache-log/access.log'

        # Make directory
        self._mkdir(dir)

        # Download
        try:
            if self._download_file_with_monitor(logs_url, dir) is None:
                raise ValueError
        except Exception as exc:
            self.logger.critical('Unable to download {0}: {1}'.format(logs_url, exc))

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
                        choices=['all', 'edgar', 'svds', 'secrepo', 'almhuette_raith'])
    parser.add_argument('--verbose', '-v', help='Logging verbosity level',
                        type=int, default=0)
    parser.add_argument('--destination', '-d', help='Download destination',
                        type=str, default=os.getcwd())
    args = parser.parse_args()

    downloader = DatasetDownloader(**vars(args))

    if args.dataset == 'all':
        downloader.download_edgar()
        downloader.download_svds()
        downloader.download_secrepo()
        downloader.download_almhuette_raith()
    elif args.dataset == 'edgar':
        downloader.download_edgar()
    elif args.dataset == 'svds':
        downloader.download_svds()
    elif args.dataset == 'secrepo':
        downloader.download_secrepo()
    elif args.dataset == 'almhuette_raith':
        downloader.download_almhuette_raith()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
