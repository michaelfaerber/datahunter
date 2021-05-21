# Dataset Search UI

In this repository, we provide the source code for the UI of our [dataset search engine](https://github.com/michaelfaerber/dataset-search).

## Setup

The dataset search UI is based on Django and SQLite.

### Create Python virtual environment

We recommend to setup a Python virtual environment with SQLite and Django.
Detailled instructions follow soon.

### Run

To run the UI on the Django server, execute within the main directory
<code>python manage.py runserver 0:{{PORT}}</code>

Typical port numbers are 8000, 8080 or 9000. If you do not specify any port number, the port 8000 will be used automatically. To access the application go to https://localhost:{{PORT}}. In case you are running the application on a remote server, replace _localhost_ by the remote server's IP address.

### Migrations

Usually, you need to migrate all changes such that they are reflected in your database. For this reason, run the following commands:

1. python manage.py makemigrations 
2. python manage.py migrate 
