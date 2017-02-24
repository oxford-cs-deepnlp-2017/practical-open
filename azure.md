# Microsoft Azure Sponsorship

<img src="https://rawgit.com/oxford-cs-deepnlp-2017/practical-open/master/doc/azure.svg" width="50%" />

We would like to thank Microsoft Azure Sponsorship 2 for their generous donation of GPU computational resources for this course, as well as, Michael Thomas, Andrew Webber and Lee Stott for their precious help.


### Specifications
Each student is given a VM instance with the following specifications:

- Name: Standard NC6 (East US)
- CPU: 6 cores
- RAM: 56 GB
- GPU: NVIDIA Tesla K80


### Connect to your server
By this point you should have received an IP and a password in your college email. Using the ip and password provided you can connect to your server using `ssh deepnlp2017@<your server ip>`

#### Session managment
`tmux` allows access multiple separate terminal sessions inside a remote terminal session. Using ```tmux``` you can keep sessions alive in the background (e.g. training your network) even when disconnecting, and reconnect to them the next time you login.
Basic shortcuts:

- list sessions
`tmux ls`

- detach from session
`ctrl+b - d`

- change name of session
`ctrl+b - shift+4`

- reconnect to session
`tmux attach -t <session_name>`



#### Install Cuda 8.0

```
sudo apt-get install build-essential
cd ~
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
chmod +x cuda_8.0.61_375.26_linux-run
sudo ./cuda_8.0.61_375.26_linux-run
```

Answer the questions with y(es) or enter (default settings) for paths. To confirm successful installation run `nvidia-smi`.

#### Install CuDNN 5.1

```
cd ~
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz
sudo ldconfig
```

#### Add in references
append the following lines to your `~/.bashrc`. You can edit it with `nano ~/.bashrc`.

```
export CUDA_HOME=/usr/local/cuda-8.0
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

#### Install framework
You are ready to install python and your favourite framework:

- PyTorch
- Tensorflow
- CNTK

Instructions of how to set up an iPython notebook server remotely can be found [here](https://ipython.org/ipython-doc/3/notebook/public_server.html).

#### Troubleshooting
- ensure uninstallation of previous NVIDIA drivers `sudo apt-get remove --purge nvidia-*`

#### Support
For further information for Univerisity of Oxford students please contact `iassael@gmail.com`.