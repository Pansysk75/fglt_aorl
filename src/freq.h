#ifndef FREQ_H
#define FREQ_H

#include <stddef.h>
#include <stdlib.h>

#include "csx.h"

struct FREQ {
  /* num of vertices */
  size_t v;

  size_t *s0;
  size_t *s1;
  size_t *s2;
  size_t *s3;
  size_t *s4;
};

typedef struct FREQ *freq;

freq freq_new(size_t number_of_vertices);

void freq_free(freq f);

void freq_device_to_host(freq f, freq f_d);

freq freq_calc(csx A);

void freq_print(freq f);

#endif // !FREQ_H
