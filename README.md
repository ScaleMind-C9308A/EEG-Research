# EEG-Research

Use the dataset from 2 sources:
- S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909
[https://github.com/perceivelab/eeg_visual_classification](https://github.com/perceivelab/eeg_visual_classification)

## I. Setup Guideline: 

- Follow the commands in Git Bash (Windows) or Terminal (Linux/Mac)

### *Setup repo for first-time clone:

- Create .env folder and install neccessary packages 

```
virtualenv .env 
source .env/Scripts/activate # For Windows
source .env/bin/activate # For Linux
pip install -r requirements.txt
```

### *Sync with submodule repo

https://stackoverflow.com/questions/18770545/why-is-my-git-submodule-head-detached-from-master/55570998#55570998

In order to sync all of the submodules:

```bash
# cd back to project root
git submodule update --remote --merge
```

### *How to ssh into the server 
- Step 1: Install tailscale https://tailscale.com/              
- Step 2: In terminal (preferably Git Bash): `ssh exx@100.98.174.90`
- Step 3: Login tailscale with google account scalemindserver@gmail.com
- Step 4: Enjoy

### *\[SSH Tools] Tmux (enable background running)

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

- In your local computer's terminal, run:

    - In the first terminal:

```bash
# SSH into the server to run jupyter notebook
ssh <server_name>@<server_ip_address>
tmux a -t eeg # <Or Ctrl-T s> Choose tmux session name "eeg"
```

- In the second terminal:

```bash
# Local port forwarding
ssh -NL 8888:localhost:8888 <server_name>@<server_ip_address>
```

- Copy the jupyter notebook's address in the first terminal 
- Open your browser and paste the copied address to address bar

### Tmux useful commands

- prefix: 
    - default: Ctrl-B
    - server: Ctrl-T

#### Manage sessions
- Change session: `prefix s` -> choose session -> enter
- New session: `tmux new -s <s_name>`
- Detach: `tmux detach` / `prefix d`
- Attach: `tmux a -t <s_name>`
- Rename session: `prefix $`

#### Manage windows
- New window: `prefix c`
- Next window: `prefix n`
- Prev window: `prefix p`

#### Manage panes
- Split window vertically: `prefix "`
- Split window horizontally: `prefix %`
- Navigate panes: `prefix <Arrows>`

