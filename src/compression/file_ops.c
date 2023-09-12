#include "compression/file_ops.h"

size_t fsize_orDie(const char *filename) {
  struct stat st;
  if (stat(filename, &st) != 0) {
    /* error */
    perror(filename);
    exit(ERROR_fsize);
  }

  off_t const fileSize = st.st_size;
  size_t const size = (size_t)fileSize;
  /* 1. fileSize should be non-negative,
   * 2. if off_t -> size_t type conversion results in discrepancy,
   *    the file size is too large for type size_t.
   */
  if ((fileSize < 0) || (fileSize != (off_t)size)) {
    fprintf(stderr, "%s : filesize too large \n", filename);
    exit(ERROR_largeFile);
  }
  return size;
}

FILE *fopen_orDie(const char *filename, const char *instruction) {
  FILE *const inFile = fopen(filename, instruction);
  if (inFile) return inFile;
  /* error */
  perror(filename);
  exit(ERROR_fopen);
}

void fclose_orDie(FILE *file) {
  if (!fclose(file)) {
    return;
  }
  /* error */
  perror("fclose");
  exit(ERROR_fclose);
}

size_t fread_orDie(void *buffer, size_t sizeToRead, FILE *file) {
  size_t const readSize = fread(buffer, 1, sizeToRead, file);
  if (readSize == sizeToRead) return readSize; /* good */
  if (feof(file)) return readSize;             /* good, reached end of file */
  /* error */
  perror("fread");
  exit(ERROR_fread);
}

size_t fwrite_orDie(const void *buffer, size_t sizeToWrite, FILE *file) {
  size_t const writtenSize = fwrite(buffer, 1, sizeToWrite, file);
  if (writtenSize == sizeToWrite) return sizeToWrite; /* good */
  /* error */
  perror("fwrite");
  exit(ERROR_fwrite);
}

void *malloc_orDie(size_t size) {
  void *const buff = malloc(size);
  if (buff) return buff;
  /* error */
  perror("malloc");
  exit(ERROR_malloc);
}

size_t loadFile_orDie(const char *fileName, void *buffer, size_t bufferSize) {
  size_t const fileSize = fsize_orDie(fileName);
  CHECK(fileSize <= bufferSize, "File too large!");

  FILE *const inFile = fopen_orDie(fileName, "rb");
  size_t const readSize = fread(buffer, 1, fileSize, inFile);
  if (readSize != (size_t)fileSize) {
    fprintf(stderr, "fread: %s : %s \n", fileName, strerror(errno));
    exit(ERROR_fread);
  }
  fclose(inFile); /* can't fail, read only */
  return fileSize;
}

void *mallocAndLoadFile_orDie(const char *fileName, size_t *bufferSize) {
  size_t const fileSize = fsize_orDie(fileName);
  *bufferSize = fileSize;
  void *const buffer = malloc_orDie(*bufferSize);
  loadFile_orDie(fileName, buffer, *bufferSize);
  return buffer;
}

void saveFile_orDie(const char *fileName, const void *buff, size_t buffSize) {
  FILE *const oFile = fopen_orDie(fileName, "wb");
  size_t const wSize = fwrite(buff, 1, buffSize, oFile);
  if (wSize != (size_t)buffSize) {
    fprintf(stderr, "fwrite: %s : %s \n", fileName, strerror(errno));
    exit(ERROR_fwrite);
  }
  if (fclose(oFile)) {
    perror(fileName);
    exit(ERROR_fclose);
  }
}
