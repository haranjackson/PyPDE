#ifndef INDEXING_H
#define INDEXING_H

#include "../types.h"

void update_inds(iVecr inds, iVecr bounds);

void update_inds(iVecr inds, int N);

int index(iVecr inds, iVecr bounds, int offset);

int index(iVecr inds, int N);

#endif