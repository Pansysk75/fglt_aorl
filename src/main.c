#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "csx.h"
#include "freq.h"

double get_elapsed_ms(struct timespec start, struct timespec end) {
  return (double)(end.tv_sec - start.tv_sec) * 1000 +
         (double)(end.tv_nsec - start.tv_nsec) / 1000000;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
    exit(1);
  }

  // printf("CILK_NWORKERS: %d\n\n", __cilkrts_get_nworkers());

  struct timespec start, end;

  /* Read mtx file */
  printf("Reading mtx file.. ");
  fflush(stdout);
  clock_gettime(CLOCK_MONOTONIC, &start);
  csx csc = csc_from_file(argv[1]);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("%fms\n", get_elapsed_ms(start, end));
#ifdef DEBUG
  csx_print(csc);
#endif /* ifdef DEBUG */
  printf("Vertices: %zu\n"
         "Edges:    %zu\n"
         "\n",
         csc->v, csc->e);

  printf("Calculating σ0-4.. ");
  fflush(stdout);
  clock_gettime(CLOCK_MONOTONIC, &start);
  freq f = freq_calc(csc);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("%fms\n", get_elapsed_ms(start, end));

#ifdef DEBUG
  freq_print(f);
#endif /* ifdef DEBUG */

  freq_free(f);
  csx_free(csc);

  return 0;
}
