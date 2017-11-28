# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import errno
import gzip
import logging.config
import os
import posixpath
import re
import shutil
import sys
import zipfile

from urllib.parse import urlsplit, unquote

import requests
import tqdm
from bs4 import BeautifulSoup

from datetime import datetime, timedelta

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
                filename = self._url2filename(url)  # filename
                with open(os.path.join(dir_name, filename), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
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
                filename = self._url2filename(url)  # filename
                total_size = (int(r.headers.get('content-length', 0)) / 1024)
                with open(os.path.join(dir_name, filename), 'wb') as f:
                    for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024), total=total_size, unit='B',
                                           unit_scale=True):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                return filename
            else:
                return None

    def download_r(self,startDate,endDate):
        self.logger.debug('Downloading R')

        # Download dir
        dir = '/home/minh/Desktop/R_download/'

        # Base URL of actual dataset
        base_url = 'http://cran-logs.rstudio.com/2017/'

        dateFormat = '%Y-%m-%d'

        startDate = datetime.strptime(startDate,dateFormat)
        endDate = datetime.strptime(endDate,dateFormat)

        URLs = []
        delta = endDate - startDate
        for i in range(delta.days+1):
            #print(datetime.strftime(startDate+timedelta(days=i),dateFormat))
            URLs.append(base_url+datetime.strftime(startDate+timedelta(days=i),dateFormat)+'-r.csv.gz')

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        future_to_filename = {executor.submit(self._download_file, url, dir): url for url in URLs}

        self._mkdir(dir)

        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_filename), total=len(future_to_filename)):
            url = future_to_filename[future]
            try:
                filename = future.result()
                if filename is None:
                    raise ValueError
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                zip_path = os.path.join(dir, filename)

                os.rename(os.path.join(dir, filename), os.path.join(dir, filename))
            except Exception as exc:
                self.logger.critical('DOWNLOAD_FAILED {0}: {1}'.format(url, exc))

        # cleanup
        executor.shutdown(True)

    def download_edgar(self):
        self.logger.debug('Downloading EDGAR')

        # Download dir, a subdir with year will be created
        #dir = os.path.join(self.destination, 'edgar')
        #temp_dir = '/tmp/.edgar'
        temp_dir = '/home/minh/Desktop/edgar/'
        dir = temp_dir

        # List of all log-files
        list_url = 'https://www.sec.gov/files/EDGAR_LogFileData_thru_Dec2016.html'
        # Base URL of actual dataset
        base_url = 'www.sec.gov/dera/data/Public-EDGAR-log-file-data/'

        year = input('Enter the year [2003 ~ 2016]: ')
        if int(year) in range(2003, 2017):
            # Make directory
            dir = os.path.join(dir, year)
            self._mkdir(dir)
            self._mkdir(temp_dir)

            with requests.get(list_url) as r:
                soup = BeautifulSoup(r.text, 'html.parser')
            urls = [x for x in soup.body.string.split('\r\n') if x.startswith(year, len(base_url))]

            # Schedule download
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
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

                    """with zipfile.ZipFile(zip_path) as logzip:
                        with logzip.open(csv_filename, 'r') as f_in, \
                                open(os.path.join(dir, csv_filename), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(zip_path)"""

                    os.rename(os.path.join(temp_dir,filename),os.path.join(dir,filename))
                except Exception as exc:
                    self.logger.critical('DOWNLOAD_FAILED {0}: {1}'.format(url, exc))

            # cleanup
            executor.shutdown(True)
            #shutil.rmtree(temp_dir)

    def download_svds(self):
        self.logger.debug('Downloading SVDS')

        # Directory & URLs
        #dir = os.path.join(self.destination, 'svds')
        dir = '/home/minh/Desktop/svds'
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
            self.logger.critical('DOWNLOAD_FAILED {0}: {1}'.format(logs_url, exc))

    def download_secrepo2016(self):
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
                self.logger.critical('DOWNLOAD_FAILED {0}: {1}'.format(url, exc))

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
            self.logger.critical('DOWNLOAD_FAILED {0}: {1}'.format(logs_url, exc))

    def download_maccdc2012(self):
        self.logger.debug('Downloading MACCDC2012')

        # Directory & URLs
        #dir = os.path.join(self.destination, 'maccdc2012')
        #temp_dir = '/tmp/.maccdc2012'
        temp_dir = 'home/minh/Desktop/maccdc2012'
        dir = temp_dir
        logs_url = 'http://www.secrepo.com/maccdc2012/http.log.gz'

        # Make directory
        self._mkdir(dir)
        self._mkdir(temp_dir)

        # Download
        try:
            filename = self._download_file_with_monitor(logs_url, temp_dir)
            if filename is None:
                raise ValueError
            temp_path = os.path.join(temp_dir, filename)

            with gzip.open(temp_path, 'rb') as f_in, \
                    open(os.path.join(dir, os.path.splitext(filename)[0]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(temp_path)
        except Exception as exc:
            self.logger.critical('DOWNLOAD_FAILED {0}: {1}'.format(logs_url, exc))

        # cleanup
        #shutil.rmtree(temp_dir)

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


def harvest_log(log_path):
    with open(log_path, 'r') as f:
        for line in f:
            try:
                line_parse = line.split()
                url_index = line_parse.index('DOWNLOAD_FAILED')
                print(line_parse[url_index + 1][:-1])
            except:
                print("Unable to parse: {}".format(line))


def main():
    downloader = DatasetDownloader()
    downloader.download_r('2017-08-01','2017-10-31')

    """parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset(s) to download', type=str,
                        choices=['all', 'edgar', 'svds', 'secrepo2016', 'almhuette_raith', 'maccdc2012'],
                        nargs=1)
    parser.add_argument('--harvest', help='Harvests failed URLs from logfile',
                        type=str, nargs=1)
    parser.add_argument('--verbose', '-v', help='Logging verbosity level',
                        type=int, default=0)
    parser.add_argument('--destination', '-d', help='Download destination',
                        type=str, default=os.getcwd())
    args = parser.parse_args()

    if args.harvest:
        harvest_log(args.harvest[0])
    elif args.dataset:
        downloader = DatasetDownloader(**vars(args))
        if args.dataset[0] == 'all':
            downloader.download_edgar()
            downloader.download_svds()
            downloader.download_secrepo2016()
            downloader.download_almhuette_raith()
            downloader.download_maccdc2012()
        elif args.dataset[0] == 'edgar':
            downloader.download_edgar()
        elif args.dataset[0] == 'svds':
            downloader.download_svds()
        elif args.dataset[0] == 'secrepo2016':
            downloader.download_secrepo2016()
        elif args.dataset[0] == 'almhuette_raith':
            downloader.download_almhuette_raith()
        elif args.dataset[0] == 'maccdc2012':
            downloader.download_maccdc2012()
    else:
        parser.print_help()"""


if __name__ == '__main__':
    main()
