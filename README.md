# DataHunter / Dataset Search UI

In this repository, we provide the source code for the UI of our dataset search engine [DataHunter](https://github.com/michaelfaerber/dataset-search).

## Abstract

The number of datasets is steadily rising, making it increasingly difficult for researchers and practitioners in the various scientific disciplines to be aware of all datasets, particularly of the most relevant datasets for a given research problem. To this end, dataset search engines have been proposed. However, they are based on the users' keywords and thus have difficulties in determining precisely fitting datasets for complex research problems. In this paper, we propose the system at http://data-hunter.io that recommends suitable datasets to users based on given research problem descriptions. It is based on fastText for the text representation and text classification, the Data Set Knowledge Graph (DSKG) with metadata about almost 1,700 unique datasets, as well as 88,000 paper abstracts as research problem descriptions for training the model. Overall, our system demonstrates that recommending datasets facilitates data provisioning and reuse according to the FAIR principles and that dataset recommendation is a promising future research direction.

## Demo 

http://data-hunter.io

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

## Contact

The system has been designed and implemented by Michael Färber and Ann-Kathrin Leisinger. Feel free to reach out to us:

[Michael Färber](https://sites.google.com/view/michaelfaerber), michael.faerber@kit&#46;edu

## How to Cite

Please cite our [paper](https://aifb.kit.edu/images/8/89/DataHunter_RecSys2021.pdf) (published at RecSys'21) as follows:
```
@inproceedings{Faerber2021RecSys,
  author    = {Michael F{\"{a}}rber and
               Ann-Kathrin Leisinger},
  title     = "{DataHunter: A System for Finding Datasets Based on Scientific Problem Descriptions}",
  booktitle = "{Proceedings of the 15th ACM Conference on Recommender Systems}",
  location  = "{Amsterdam, The Netherlands}",
  year      = {2021}
}
```
