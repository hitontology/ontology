FROM alpine
RUN apk add raptor2
WORKDIR /ontology
COPY . .
RUN ./build
VOLUME /ontology/dist
