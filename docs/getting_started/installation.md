# Installation


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
then create a super user for admin page
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