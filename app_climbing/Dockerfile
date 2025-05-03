FROM chromadb/chroma:latest

ENV CHROMA_SERVER_NOFILE=25000
ENV CHROMA_PERSIST_DIRECTORY=/chromadb
ENV PORT=8000

WORKDIR /chromadb

EXPOSE 8000

ENTRYPOINT ["chroma"]
CMD ["run", "--path", "/chromadb", "--port", "8000", "--host", "0.0.0.0"]