FROM alpine
RUN apk add raptor2
WORKDIR /src
COPY . .
RUN ./build
VOLUME /dist
