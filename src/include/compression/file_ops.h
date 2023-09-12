#ifndef FTRL_FFM_FILE_OPS_H
#define FTRL_FFM_FILE_OPS_H

#include <errno.h>     // errno
#include <stdio.h>     // fprintf, perror, fopen, etc.
#include <stdlib.h>    // malloc, free, exit
#include <string.h>    // strerror
#include <sys/stat.h>  // stat
#include <zstd.h>

typedef enum {
  ERROR_fsize = 1,
  ERROR_fopen = 2,
  ERROR_fclose = 3,
  ERROR_fread = 4,
  ERROR_fwrite = 5,
  ERROR_loadFile = 6,
  ERROR_saveFile = 7,
  ERROR_malloc = 8,
  ERROR_largeFile = 9,
} COMMON_ErrorCode;

#define CHECK(cond, ...)                                                      \
  do {                                                                        \
    if (!(cond)) {                                                            \
      fprintf(stderr, "%s:%d CHECK(%s) failed: ", __FILE__, __LINE__, #cond); \
      fprintf(stderr, "" __VA_ARGS__);                                        \
      fprintf(stderr, "\n");                                                  \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define CHECK_ZSTD(fn)                                       \
  do {                                                       \
    size_t const err = (fn);                                 \
    CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err)); \
  } while (0)

/*! fsize_orDie() :
 * Get the size of a given file path.
 *
 * @return The size of a given file path.
 */
size_t fsize_orDie(const char *filename);

/*! fopen_orDie() :
 * Open a file using given file path and open option.
 *
 * @return If successful this function will return a FILE pointer to an
 * opened file otherwise it sends an error to stderr and exits.
 */
FILE *fopen_orDie(const char *filename, const char *instruction);

/*! fclose_orDie() :
 * Close an opened file using given FILE pointer.
 */
void fclose_orDie(FILE *file);

/*! fread_orDie() :
 *
 * Read sizeToRead bytes from a given file, storing them at the
 * location given by buffer.
 *
 * @return The number of bytes read.
 */
size_t fread_orDie(void *buffer, size_t sizeToRead, FILE *file);

/*! fwrite_orDie() :
 *
 * Write sizeToWrite bytes to a file pointed to by file, obtaining
 * them from a location given by buffer.
 *
 * Note: This function will send an error to stderr and exit if it
 * cannot write data to the given file pointer.
 *
 * @return The number of bytes written.
 */
size_t fwrite_orDie(const void *buffer, size_t sizeToWrite, FILE *file);

/*! malloc_orDie() :
 * Allocate memory.
 *
 * @return If successful this function returns a pointer to allo-
 * cated memory.  If there is an error, this function will send that
 * error to stderr and exit.
 */
void *malloc_orDie(size_t size);

/*! loadFile_orDie() :
 * load file into buffer (memory).
 *
 * Note: This function will send an error to stderr and exit if it
 * cannot read data from the given file path.
 *
 * @return If successful this function will load file into buffer and
 * return file size, otherwise it will printout an error to stderr and exit.
 */
size_t loadFile_orDie(const char *fileName, void *buffer, size_t bufferSize);

/*! mallocAndLoadFile_orDie() :
 * allocate memory buffer and then load file into it.
 *
 * Note: This function will send an error to stderr and exit if memory allocation
 * fails or it cannot read data from the given file path.
 *
 * @return If successful this function will return buffer and bufferSize(=fileSize),
 * otherwise it will printout an error to stderr and exit.
 */
void *mallocAndLoadFile_orDie(const char *fileName, size_t *bufferSize);

/*! saveFile_orDie() :
 *
 * Save buffSize bytes to a given file path, obtaining them from a location pointed
 * to by buff.
 *
 * Note: This function will send an error to stderr and exit if it
 * cannot write to a given file.
 */
void saveFile_orDie(const char *fileName, const void *buff, size_t buffSize);

#endif  // FTRL_FFM_FILE_OPS_H
