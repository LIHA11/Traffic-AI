# SEE https://oworkspace.sharepoint.com/sites/KM/OOCLWIKI/Pages/Available-Docker-Base-Image-List.aspx to prepare your microservice image Dockerfile

FROM artifact-docker-base-image-local.digital.oocl.com/common/python-oocl:3.11.12-20250430

WORKDIR /usr/src/app

COPY --chown=node:node . .
RUN pip install --user --default-timeout=180 -r requirements.txt 

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

### for sonatype scan
RUN pip freeze --all > /tmp/requirements.txt  
###

### Remove pip after install required libraries
USER root
RUN pip uninstall -y pip
USER node
###

EXPOSE 8080

ENTRYPOINT ["python", "app.py"] 