Dataset Downloader
-----------------
To download a dataset, run:
```bash
(venv)$ python download_datasets.py --dataset name_of_dataset
```

Supported values for ```name_of_dataset``` are:
- edgar: [URL](https://www.sec.gov/data/edgar-log-file-data-set.html)
- svds: [URL](https://github.com/silicon-valley-data-science/datasets/)
- secrepo2016: [URL](www.secrepo.com/self.logs/2016/)
- almhuette_raith: [URL](http://www.almhuette-raith.at/apache-log/access.log)
- maccdc2012: [URL](http://www.secrepo.com/maccdc2012/http.log.gz)

A special ```all``` value supplied as ```name_of_dataset``` will download all datasets.
You might want to grab a meal (or two or three ...) if you choose to download all datasets at once! 

By default, all datasets will be downloaded in their respective folders relative to current working directory.
The base directory (defaults to ```os.getcwd()```) can be changed by using ```--destination``` argument.

```bash
(venv)$ python download_datasets.py --dataset all --destination /path/to/destination
```

:warning: **CAUTION** <br/>
It is highly likely that some files will fail to download!
In such cases, the failed URLs will be logged in the file named ```dataset-downloader.log```
You can harvest the failed URLs by using
```bash
(venv)$ python download_datasets.py --harvest /path/to/dataset-downloader.log
```
and then (**highly recommended**) download them separately using external tools (e.g. ```aria2c```)
```bash
$ seq 1000 | parallel -j1 aria2c -i failed_urls.list -c
```

For help, use ```--help```
```bash
(venv)$ python download_datasets.py --help
```
or create an [issue](https://github.com/minhresl/Ensemble/issues)