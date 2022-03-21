FROM alpine
RUN apk add raptor2
WORKDIR /ontology
COPY . .
RUN ./build && rm dist/all.ttl
VOLUME /ontology/dist
