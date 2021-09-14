#!/bin/bash
export GPN_DB_PORT=27017
export GPN_DB_HOST=localhost
export GPN_DB_NAME=gpn

export GPN_PARSER_URL=http://localhost:8888
export GPN_ADMIN_GROUP=CN=Admin,OU=department2,OU=department1,OU=Company,DC=company,DC=loc
export GPN_AUDIT_GROUP=CN=Group1,OU=Company,DC=company,DC=loc
export GPN_EVENT_GROUP=CN=Group2,OU=Company,DC=company,DC=loc

printenv

#mongod --config /usr/local/etc/mongod.conf
mongod --config /opt/homebrew/etc/mongod.conf