# GeneFromAPI
---
GeneFromAPI is website for gene searching with open API. It used Harmonizome and OpenTarget API. Especially only CTD Gene-Disease Associations Dataset was used from Harmonizome.


## Using the GeneFromAPI `Docker` image
Now, only docker is available to use GeneFromAPI.
If docker is not installed, please install docker at first. Installation commands for docker are written in `docker_install.sh`. When this script is not worked so that docker is not installed, please check the comment of that script. If the recommended command is not work, please install the docker by another way.

Please clone the git repository to your workspace.
```
git clone https://github.com/khsjh/GeneFromAPI.git
```

And build docker with `gene_explorer` directory
```
docker compose -f docker_compose.yml up -d --build # Docker build
```

You can check whether docker is built successfully. If "UP" is displayed in the `STATUS` column, the work is done.
```
docker compose -f docker_compose.yml ps # 확인용

# output
AME                IMAGE               COMMAND                  SERVICE   CREATED          STATUS          PORTS
gene_explorer_api   gene_explorer-api   "uvicorn main:app --…"   api       55 minutes ago   Up 55 minutes   8000/tcp
gene_explorer_web   nginx:alpine        "/docker-entrypoint.…"   web       55 minutes ago   Up 49 minutes   0.0.0.0:8080->80/tcp, [::]:8080->80/tcp
```
Now, you can access the GeneFromAPI with your local address

```
http://localhost:8080
```

You can terminate the docker by below command.
```
docker compose -f docker_compose.yml down # Docker 종료
```

## Reference
Diamant I, Clarke DJB, Evangelista JE, Lingam N, Ma'ayan A. Harmonizome 3.0: integrated knowledge about genes and proteins from diverse multi-omics resources. Nucleic Acids Res. 2024 Nov 20. pii: 53(1):D1016-D1028.

Buniello, A. et al. (2025). Open Targets Platform: facilitating therapeutic hypotheses building in drug discovery. Nucleic Acids Research.
