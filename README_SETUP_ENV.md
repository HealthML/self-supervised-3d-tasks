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
```bash
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```

If this step fails with a permission denied 
then the file is probably already downloaded into `/tmp`.
Just proceed with the next step (you can also check for this via `ls`)

##### Install Anaconda
```bash
bash Anaconda3-2019.10-Linux-x86_64.sh
```

Make sure you install this as a **local** install only (which should be the default as well).

##### Activate Conda
```bash
source ~/.bashrc
``` 

##### Create Anaconda Environments

You can create Anaconda environments with the conda create command. For example, a Python 3 environment named my_env can be created with the following command:

For the later steps you have to do this _once_ on the server while connected via ssh.
```bash
conda create --name conda-env python=3.6
```

##### Create the folder your Project Instance will live in

You _should_ create a folder where pycharm will map your project to. 
```bash
cd ~ 
mkdir self-supervised-3d-taks
```

(The name for the folder in the `mkdir` command is your choice)


## Configure PyCharm SSH Interpreter
After you have installed all required software tools, you can configure your PyCharm Project interpreter. 
Go to `File --> Settings... --> Project: ... --> Project Interpreter` and 
click on the small icon on the top right. 
Here you can add a project interpreter. 

In the `Add Python Interpreter` dialog you can add a SSH Interpreter.
Enter `vm-mpws2019cl1.dhclab.i.hpi.de` as a host and your username `Firstname.Lastname`. 
In the next dialog it will ask for your password or ssh key. 
Afterwards you have to select an interpreter on the remote server and the mapping directory. 
- **Mapping directory**: 
    Pycharm will try to use a folder in `/tmp`, change this to the folder you made earlier on the server.
    You can find your directory under `/home/<Firstname.Lastname>/<your folder name>`
- **Interpreter**: 
    Please select a project interpreter from your conda environments, e.g. `~/Anaconda3/conda/envs/<your env>/bin/python3` 

After you have applied these changes, PyCharm will upload the files to the server. 

##  Run Code
### Setup conda env
Now you have to set up your conda environment dependencies. 
For that you have to go to your mapping directory on the server and run 

```bash
 conda activate conda-env
 conda env update -f env_all_platforms.yml
```
 
### Activate / Deactivate Conda Env
Activate:

```bash
 conda activate conda-env
```

Deactivate:
```bash
 conda deactivate 
```

### Execute Code
To run the code,  navigate to your mapping directory and execute the shell in your activated environment.
```bash
sh ./config/jigsaw/ukb3d.sh 
```


