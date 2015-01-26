Initial (successful!!) setup for running on EC2 AMI: 
ami-6238470a

### Miscellaneous setup

Prepare a volume: 
    sudo mkfs -t ext4 /dev/xvdf

Mount volume:
    sudo mkdir ~/data
    sudo chmod a+w ~/data
    sudo mount /dev/xvdf ~/data

Security group settings (from [here](https://gist.github.com/iamatypeofwalrus/5183133)

> You should have rules for SSH(22): 0.0.0.0/0, HTTPS(443): 0.0.0.0/0, and 8888: 0.0.0.0/0

(Use `Custom TCP Rule` for the 8888)

IPython notebook setup (from 
[here](http://ipython.org/ipython-doc/dev/notebook/public_server.html#running-a-public-notebook-server))

    ipython profile create nbserver
    ipython
    from IPython.lib import passwd
    passwd()

Copy the hashed password. Then, 

    cd anaconda
    !mkdir certificates
    cd certificates
    !openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem

Edit `~/.ipython/profile_nbserver/ipython_notebook_config.py`:
    c.NotebookApp.certfile = u'/home/ubuntu/anaconda/certificates/mycert.pem'
    c.NotebookApp.notebook_dir = u'/home/ubuntu/notebook'
    c.NotebookApp.ip = '*'
    c.NotebookApp.open_browser = False
    c.NotebookApp.password = u'<hashed password goes here>'
    c.NotebookApp.port = 8888

Run with
    ipython notebook --profile=nbserver
