Initial setup for running on EC2 AMI: 
ami-6238470a


### TODO

Add code that uploads models to s3

aws cli install: 

    sudo pip install awscli
    aws configure

copy using:

    aws s3 cp mlp1.lua s3://johndun.aws.bucket/