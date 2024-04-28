# ARTA: Automatic Requirement Testability Analyzer

![ARTA_logo](docs/figs/logo.png)


Software testability is the propensity of software artifact to reveal its existing defects. 
Software requirements are crucial artifacts in developing software. Requirements specifications are used in both the functional and acceptance testing to ensure that a program meets its requirements. 
A testable requirement increases the effectiveness of testing while decreasing the cost and time. In this project, we define requirement testability in terms of requirements smells and propose a measuring method.
More information are available in the [project documentation website](https://m-zakeri.github.io/ARTA).


## Demo

ARTA is a research project at [IUST Reverse Engineering Laboratory](http://reverse.iust.ac.ir/).
An online demo of ARTA is available on 
[http://arta.iust-parsa.ir/](http://arta.iust-parsa.ir/).
You can login with following credential:

 * Demo username: User
 * Demo password: arta@IUST

and watch the examples requirements.  


### ARTA requirement analyzer module

![ARTA Demo 1](./docs/figs/ARTA_screenshot1.png)


### ARTA requirement smell labeling module

![ARTA Demo 2](./docs/figs/ARTA_screenshot2.png)


## Getting started

### Prerequisites

#### MySql

MySQL version 5.6 is required.

##### install MySql on windows
Download and install using this [link](https://dev.mysql.com/downloads/installer/).

##### install MySql on ubuntu
Following instructions on this [link](https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-20-04).

#### Python
since project based on python, **python version 3.8+** needed.

download python 3.8 for windows from [here](https://www.python.org/downloads/).


### Install 

#### Create a virtual environment
First, install virtualenv

```bash
python3 -m pip install virtualenv
cd project_directory
virtualenv env 
```

Activate virtual environment using:

```bash
source env/bin/activate
```


#### Install project's requirements
Run
```bash
pip install -r requirements.txt
```
then install necessary packages for textBlob
```bash
python -m textblob.download_corpora
```

#### Necessary packages on ubuntu

Install necessary packages for mysqlclient on ubuntu using

```bash
sudo apt install python3-dev default-libmysqlclient-dev poppler-utils
```

#### Project configuration
First, create a mysql database for the project

then create file called local_settings.py in Smella folder

```bash
vim Smella/local_settings.py
```

write following code
```python
DB_USER = 'database username'
DB_PASS = 'database password'
DB_HOST = 'database host'
DB_NAME = 'database name'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

```
then apply migration using
```bash
python manage.py makemigrations
python manage.py migrate
```

then create a super-user for admin page

```bash
python manage.py createsuperuser
```

until the ui fully written, create requirements and projects using admin page "/admin/"


#### Running Project

run project using

```bash
python manage.py runserver
```


#### Insert Dataset

insert dataset the using.
```bash
python manage.py insert_date <location of json file> <a project name for this dataset>
```

## Used Datasets
- [Dataset 1 (available in pdf,doc,xml formats - xml versions used)](http://fmt.isti.cnr.it/nlreqdataset/)
  
- [Dataset 2 (available in txt format - needs vpn)](https://www.kaggle.com/iamsouvik/software-requirements-dataset/data)


## Publication(s)

* [1] **Zakeri-Nasrabadi, M.**, & Parsa, S. (2024). **Natural language requirements testability measurement based on requirement smells**. Neural Computing and Applications. [https://doi.org/10.1007/s00521-024-09730-x](https://doi.org/10.1007/s00521-024-09730-x)

## News

* **2024-04-28:** The ARTA (automated requirements testability analyzer) tool's full implementation is now publicly available.

* **2021-10-20:** Initial/partial release. The full version of source code will be available as soon as the relevant paper(s) are published.


## Read more

Visit project website at [https://m-zakeri.github.io/ARTA](https://m-zakeri.github.io/ARTA)

