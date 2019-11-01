# Set Up Environment
This set up tutorial explains how to deploy with pycharm on the remote server `vm-mpws2019cl1.dhclab.i.hpi.de`.  

## Install Required Software Tools

### Local

#### Pycharm Professional
You can get a PyCharm Professional license with your HPI e-mail address.   

### Remote
You can use ssh connection to work on remote server.

#### Anaconda 
You need anaconda for managing your virtual environments on the machine:
##### Download Anaconda
```
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```
##### Install Anaconda
```
bash Anaconda3-2019.10-Linux-x86_64.sh
```

##### Activate Conda
```
source ~/.bashrc
``` 

##### Create Anaconda Environments

You can create Anaconda environments with the conda create command. For example, a Python 3 environment named my_env can be created with the following command:
```
conda create --name conda-env python=3.6
```


## Configure PyCharm SSH Interpreter
After you have installed all required software tools, you can configure your PyCharm Project interpreter. Go to `File --> Settings... --> Project: ... --> Project Interpreter` and click on the small icon on the top right. Here you can add a project interpreter. 

In the `Add Python Interpreter` dialog you can add a SSH Interpreter. Enter `vm-mpws2019cl1.dhclab.i.hpi.de` as a host and your username `Firstname.Lastname`. In the next dialog it will ask for your password or ssh key. Afterwards you have to select an interpreter on the remote server and the mapping directory. 
- **Mapping directory**: Pycharm will automatically update the project in the mapping directory directory.
- **Interpreter**: Please select a project interpreter from your conda environments, e.g. `~/Anaconda3/conda/envs/<your env>/bin/python3.6` 

After you have applied these changes, PyCharm will upload the files to the server. 

##  Run Code
### Setup conda env
Now you have to set up your conda environment. For that you have to go to your mapping directory and run 

```
 conda activate conda-env
 conda env update -f env_all_platforms.yml
```
 
### Activate / Deactivate Conda Env
Activate:

```
 conda activate conda-env
```

Deactivate:
```
 conda deactivate 
```

### Execute Code
To run the code,  navigate to your mapping directory and execute the shell in your activated environment.
```
sh ./config/jigsaw/ukb3d.sh 
```


