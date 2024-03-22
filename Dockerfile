# syntax=docker/dockerfile:1
FROM alpine as builder
RUN apk add raptor2
WORKDIR /ontology   
COPY . .
RUN ./scripts/combine && rm dist/all.ttl

# "from scratch" causes "no command specified"
FROM busybox
COPY --link --from=builder /ontology/dist /ontology/dist
WORKDIR /ontology/dist
VOLUME /ontology/dist
