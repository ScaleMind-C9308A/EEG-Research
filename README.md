# EEG-Research

## Setup Guideline

- Follow the commands in Git Bash (Windows) or Terminal (Linux/Mac)

### Setup repo for first-time clone:

- Create .env folder and install neccessary packages 

```
virtualenv .env 
source .env/Scripts/activate # For Windows
source .env/bin/activate # For Linux
pip install -r requirements.txt
```

### How to ssh into the server (Managed with tmux sessions)

#### If the server has no tmux sessions and jupyter notebook is not running

- In your local computer's terminal, run:

    - In the first terminal:

```bash
# SSH into the server to run jupyter notebook
ssh <server_name>@<server_ip_address>
tmux new -s eeg # Create new tmux session named "eeg"
cd <cloned_repo>
jupyter notebook --no-browser --port 8888
tmux detach # <Or Ctrl-T d> Detach session, no worry that the session is still running in background
```

    - In the second terminal:

```bash
# Local port forwarding
ssh -NL 8888:localhost:8888 <server_name>@<server_ip_address>
```

- Copy the jupyter notebook's address in the first terminal 
- Open your browser and paste the copied address to address bar

#### If the server already has tmux sessions (jupyter is running)
- The server



