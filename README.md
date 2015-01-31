Initial setup for running on EC2 AMI: 
ami-6238470a

    ssh -i cuda-private-key.pem ubuntu@<ip>
    git clone https://github.com/johndun/kaggle-plankton
    cd kaggle-plankton
    sudo pip install awscli
    aws configure
    sudo apt-get install r-base
    bash initial_setup.sh
    Rscript --vanilla initial_data_processing.r

