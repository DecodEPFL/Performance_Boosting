#!/usr/bin/env bash
set -euo pipefail

: "${GASPAR:?Set GASPAR to your EPFL GASPAR username.}"
: "${LDAP_UID:?Set LDAP_UID to your EPFL UID.}"
: "${LDAP_GID:?Set LDAP_GID to your EPFL GID.}"
: "${PROJECT:?Set PROJECT to your Harbor project name.}"

IMAGE="${IMAGE:-performance-boosting}"
TAG="${TAG:-v1.0}"
LDAP_GROUPNAME="${LDAP_GROUPNAME:-SCI-STI-GFT}"
PLATFORM="${PLATFORM:-linux/amd64}"
REGISTRY="${REGISTRY:-registry.rcp.epfl.ch}"
IMAGE_URI="${REGISTRY}/${PROJECT}/${IMAGE}:${TAG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Would build and push ${IMAGE_URI}"
  echo "docker login -u ${GASPAR} ${REGISTRY}"
  echo "docker build --platform ${PLATFORM} --file DockerfileRCP --tag ${IMAGE_URI} ..."
  echo "docker push ${IMAGE_URI}"
  exit 0
fi

echo "Logging in to ${REGISTRY} as ${GASPAR}..."
docker login -u "${GASPAR}" "${REGISTRY}"

echo "Building ${IMAGE_URI}..."
docker build \
  --platform "${PLATFORM}" \
  --file DockerfileRCP \
  --tag "${IMAGE_URI}" \
  --build-arg "LDAP_GROUPNAME=${LDAP_GROUPNAME}" \
  --build-arg "LDAP_GID=${LDAP_GID}" \
  --build-arg "LDAP_USERNAME=${GASPAR}" \
  --build-arg "LDAP_UID=${LDAP_UID}" \
  .

echo "Pushing ${IMAGE_URI}..."
docker push "${IMAGE_URI}"

echo "Done: ${IMAGE_URI}"
