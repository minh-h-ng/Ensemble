## Configure repository
Follow the instructions [here](https://cran.rstudio.com/bin/linux/ubuntu/#installation)
and configure your apt repositories for R's installation.
We have listed the packages to install below.

## Install system dependencies
```bash
$ sudo apt-get install python3-dev python-virtualenv r-base libcurl4-openssl-dev tee
```

# Create virtual environment
```bash
$ virtualenv -p python3 venv
$ source venv/bin/activate
(venv)$
```

# Install python dependencies
```bash
(venv)$ pip3 install -r requirements.txt
```