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
By this point you should have received an IP and a password in your college email. Using the ip and password provided you can connect to your server using ```
ssh deepnlp2017@<your server ip>
```

#### Session managment
```tmux``` allows access multiple separate terminal sessions inside a remote terminal session. Using ```tmux``` you can keep sessions alive in the background (e.g. training your network) even when disconnecting, and reconnect to them the next time you login.
Basic shortcuts:

- list sessions
```tmux ls```

- detach from session
```ctrl+b - d```

- change name of session
```ctrl+b - shift+4```

- reconnect to session
```tmux attach -t <session_name>```



#### Install Cuda 8.0
```
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

To confirm successful installation run ```nvidia-smi```.
#### Install CuDNN 5.1

```
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz && \
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local && \
rm cudnn-8.0-linux-x64-v5.1.tgz && \
sudo ldconfig
```


#### Install framework
You are ready to install python and your favourite framework:

- PyTorch
- Tensorflow
- CNTK

#### Support
For further information for Univerisity of Oxford students please contact ```iassael@gmail.com```.