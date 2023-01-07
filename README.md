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

### How to ssh into the server

- In your local computer's terminal, run:

    - In the first terminal:

```bash
# First terminal
ssh <server_name>@<server_ip_address>
cd <cloned_repo>
jupyter notebook --no-browser --port 8888
```
    - In the second terminal:

```bash
ssh -NL 8888:localhost:8888 <server_name>@<server_ip_address>
```

- Copy the jupyter notebook's address in the first terminal 
- Open your browser and paste the copied address to address bar




