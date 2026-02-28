# PE-SGD
Repository for paper PE-SGD: Differentially Private Deep Learning via Evolution of Gradient Subspace for Text. 

#### News!
We are happy to announce that our paper has been accepted by [ICLR-2026](https://openreview.net/forum?id=713ywmTZHv).

### Environment Installation
```bash
conda create --name pe-sgd python=3.10.13
conda deactivate
conda activate pe-sgd
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
cd PE-SGD
pip install -e .[text]
pip install "private-evolution @ git+https://github.com/microsoft/DPSDA.git"
pip install datasets==3.6.0 peft==0.16.0 trl==0.19.1 prv-accountant==0.2.0
cd ..
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,webui]"
cd ..
```

### Script Launch
Example use can be found at `./run_script.sh`. Please run the commends therein under `.` directory.