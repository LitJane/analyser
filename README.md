 

# Analyser core & nlp_tools

 

___
## Miscl. commands
- Create wheel: 
```
python setup.py bdist_wheel 
```
- Collect all wheels of the project: 
```
pip wheel -r requirements.txt --wheel-dir=tmp/wheelhouse
```
- Install collected wheels 
``` 
pip install --no-index --find-links=tmp/wheelhouse SomePackage 
```

## Assign a release tag:
1. Create a tag:
    ```                     
    > git tag -a vX.X.X -m "<release comment>"
    ```
1. Push tag:
    ```                     
    > git push origin --tags
    ```

## Usage (Windows):
1. Install Python >=3.6 and pip
1. Install ```virtualenv```( [https://virtualenv.pypa.io/en/latest/installation/]() ):
    ```
    > pip install virtualenv
    ```
1. Download ``` nemoware_analyzer-X.X.X-py3-none-any.whl ``` to working dir (e.g. ```analyser_home```)     
1. Change to the work dir:
    ```
    > cd analyser_home
    ```
1. Create virtual environment (with name ```venv```):
    ```
    > virtualenv venv
    ```
1. Activate:
    ``` 
    > .\venv\Scripts\activate
    ```
1. Install ```analyser``` with all deps:
    ```
    > pip install  .\nemoware_analyzer-X.X.X-py3-none-any.whl    
    ```
1. Run:

    ```
    > analyser_run
    ```


## Run analyzer as a service
1. Register systemd service
    ```
    > cd bin 
    > sudo ./install_service.sh 
    ```
1. Service commands
    ```
    sudo systemctl stop nemoware-analyzer.service          #To stop running service 
    sudo systemctl start nemoware-analyzer.service         #To start running service 
    sudo systemctl restart nemoware-analyzer.service       #To restart running service 
    ```
    
# CML 
### contintinous machine learning
CML is triggered only on push or pull request to `model` branch.  
refer https://github.com/nemoware/analyser/.github/workflows/cml.yaml

to run CML worker (just example, parameters may differ):

```
sudo docker run --name <ANYNAME> -d 
   -v ~/pip_cache:/pip_cache 
   -v ~/gpn:/gpn_cml 
   -e GPN_WORK_DIR=/gpn_cml   
   -e RUNNER_IDLE_TIMEOUT=18000  
   -e RUNNER_LABELS=cml,cpu   
   -e RUNNER_REPO=https://github.com/nemoware/analyser   
   -e repo_token=<personal github access token> 
   -e GPN_DB_HOST=192.168.10.36 
   -e PIP_DOWNLOAD_CACHE=/pip_cache    
   dvcorg/cml-py3
```

