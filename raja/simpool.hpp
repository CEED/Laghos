// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_SIMPOOL
#define MFEM_SIMPOOL

#include <cstdlib>
#include <strings.h>
#include <iostream>
#include <cstddef>
#include <cassert>
#if defined(MFEM_USE_CUDAUM)
#include "cuda_runtime.h"
#endif
//#warning using simpool

struct StdAllocator
{
  static inline void* allocate(std::size_t size) { return std::malloc(size); }
  static inline void  deallocate(void *ptr) { std::free(ptr); }
};

struct Allocator
{
#if defined(MFEM_USE_CUDAUM)
  static inline void *allocate(std::size_t size) {
    void *ptr;
    cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    return ptr;
  }
  static inline void deallocate(void *ptr) { cudaFree(ptr); }
#else
  static inline void* allocate(std::size_t size) { return std::malloc(size); }
  static inline void  deallocate(void *ptr) { std::free(ptr); }
#endif
};


template<class T, class MA, int NP=(1<<6)>
class FixedPoolAllocator
{
protected:
  struct Pool
  {
    unsigned char *data;
    unsigned int *avail;
    unsigned int numAvail;
    struct Pool* next;
  };

  struct Pool *pool;
  const std::size_t numPerPool;
  const std::size_t totalPoolSize;

  std::size_t numBlocks;

  void newPool(struct Pool **pnew) {
    struct Pool *p = static_cast<struct Pool *>(MA::allocate(totalPoolSize));
    p->numAvail = numPerPool;
    p->next = NULL;

    p->data  = reinterpret_cast<unsigned char *>(p) + sizeof(struct Pool);
    p->avail = reinterpret_cast<unsigned int *>(p->data + numPerPool * sizeof(T));
    for (int i = 0; i < NP; i++) p->avail[i] = -1;

    *pnew = p;
  }

  T* allocInPool(struct Pool *p) {
    if (!p->numAvail) return NULL;

    for (int i = 0; i < NP; i++) {
      int bit = ffs(p->avail[i]) - 1;
      if (bit >= 0) {
        p->avail[i] ^= 1 << bit;
        p->numAvail--;
        int entry = i * sizeof(unsigned int) * 8 + bit;
        return reinterpret_cast<T*>(p->data) + entry;
      }
    }

    return NULL;
  }

public:
  static inline FixedPoolAllocator &getInstance() {
    static FixedPoolAllocator instance;
    return instance;
  }

  FixedPoolAllocator()
    : numPerPool(NP * sizeof(unsigned int) * 8),
      totalPoolSize(sizeof(struct Pool) +
                    numPerPool * sizeof(T) + NP * sizeof(unsigned int)),
      numBlocks(0)
  { newPool(&pool); }

  ~FixedPoolAllocator() {
    for (struct Pool *curr = pool; curr; ) {
      struct Pool *next = curr->next;
      MA::deallocate(curr);
      curr = next;
    }
  }

  T* allocate() {
    T* ptr = NULL;

    struct Pool *prev = NULL;
    struct Pool *curr = pool;
    while (!ptr && curr) {
      ptr = allocInPool(curr);
      prev = curr;
      curr = curr->next;
    }

    if (!ptr) {
      newPool(&prev->next);
      ptr = allocate();
      // TODO: In this case we should reverse the linked list for optimality
    }
    else {
      numBlocks++;
    }
    return ptr;
  }

  void deallocate(T* ptr) {
    int i = 0;
    for (struct Pool *curr = pool; curr; curr = curr->next) {
      const T* start = reinterpret_cast<T*>(curr->data);
      const T* end   = reinterpret_cast<T*>(curr->data) + numPerPool;
      if ( (ptr >= start) && (ptr < end) ) {
        // indexes bits 0 - numPerPool-1
        const int indexD = ptr - reinterpret_cast<T*>(curr->data);
        const int indexI = indexD / sizeof(unsigned int) / 8;
        const int indexB = indexD % ( sizeof(unsigned int) * 8 );
#ifdef NDEBUG
        if (!((curr->avail[indexI] >> indexB) & 1))
          std::cerr << "Trying to deallocate an entry that was not marked as allocated"
                    << std::endl;
#endif
        curr->avail[indexI] ^= 1 << indexB;
        curr->numAvail++;
        numBlocks--;
        return;
      }
      i++;
    }
    std::cerr << "Could not find pointer to deallocate" << std::endl;
    throw(std::bad_alloc());
  }

  /// Return allocated size to user.
  std::size_t allocatedSize() const { return numBlocks * sizeof(T); }

  /// Return total size with internal overhead.
  std::size_t totalSize() const {
    return numPools() * totalPoolSize;
  }

  /// Return the number of pools
  std::size_t numPools() const {
    std::size_t np = 0;
    for (struct Pool *curr = pool; curr; curr = curr->next) np++;
    return np;
  }

  /// Return the pool size
  std::size_t poolSize() const { return totalPoolSize; }
};

template <class MA, class IA = StdAllocator>
class DynamicPoolAllocator
{
protected:
  struct Block
  {
    char *data;
    std::size_t size;
    bool isHead;
    Block *next;
  };

  // Allocator for the underlying data
  typedef FixedPoolAllocator<struct Block, IA, (1<<6)> BlockAlloc;
  BlockAlloc blockAllocator;

  // Start of the nodes of used and free block lists
  struct Block *usedBlocks;
  struct Block *freeBlocks;

  // Total size allocated (bytes)
  std::size_t totalBytes;

  // Allocated size (bytes)
  std::size_t allocBytes;

  // Minimum size for allocations
  std::size_t minBytes;

  // Search the list of free blocks and return a usable one if that exists, else NULL
  void findUsableBlock(struct Block *&best, struct Block *&prev, std::size_t size) {
    best = prev = NULL;
    for ( struct Block *iter = freeBlocks, *iterPrev = NULL ; iter ; iter = iter->next ) {
      if ( iter->size >= size && (!best || iter->size < best->size) ) {
        best = iter;
        prev = iterPrev;
      }
      iterPrev = iter;
    }
  }

  // Allocate a new block and add it to the list of free blocks
  void allocateBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
    const std::size_t sizeToAlloc = std::max(size, minBytes);
    curr = prev = NULL;
    void *data = NULL;

    // Allocate data
    data = MA::allocate(sizeToAlloc);
    totalBytes += sizeToAlloc;
    assert(data);

    // Find next and prev such that next->data is still smaller than data (keep ordered)
    struct Block *next;
    for ( next = freeBlocks; next && next->data < data; next = next->next ) {
      prev = next;
    }

    // Allocate the block
    curr = (struct Block *) blockAllocator.allocate();
    if (!curr) return;
    curr->data = static_cast<char *>(data);
    curr->size = sizeToAlloc;
    curr->isHead = true;
    curr->next = next;

    // Insert
    if (prev) prev->next = curr;
    else freeBlocks = curr;
  }

  void splitBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
    struct Block *next;
    if ( curr->size == size ) {
      // Keep it
      next = curr->next;
    }
    else {
      // Split the block
      std::size_t remaining = curr->size - size;
      struct Block *newBlock = (struct Block *) blockAllocator.allocate();
      if (!newBlock) return;
      newBlock->data = curr->data + size;
      newBlock->size = remaining;
      newBlock->isHead = false;
      newBlock->next = curr->next;
      next = newBlock;
      curr->size = size;
    }

    if (prev) prev->next = next;
    else freeBlocks = next;
  }

  void releaseBlock(struct Block *curr, struct Block *prev) {
    assert(curr != NULL);

    if (prev) prev->next = curr->next;
    else usedBlocks = curr->next;

    // Find location to put this block in the freeBlocks list
    prev = NULL;
    for ( struct Block *temp = freeBlocks ; temp && temp->data < curr->data ; temp = temp->next ) {
      prev = temp;
    }

    // Keep track of the successor
    struct Block *next = prev ? prev->next : freeBlocks;

    // Check if prev and curr can be merged
    if ( prev && prev->data + prev->size == curr->data && !curr->isHead ) {
      prev->size = prev->size + curr->size;
      blockAllocator.deallocate(curr); // keep data
      curr = prev;
    }
    else if (prev) {
      prev->next = curr;
    }
    else {
      freeBlocks = curr;
    }

    // Check if curr and next can be merged
    if ( next && curr->data + curr->size == next->data && !next->isHead ) {
      curr->size = curr->size + next->size;
      curr->next = next->next;
      blockAllocator.deallocate(next); // keep data
    }
    else {
      curr->next = next;
    }
  }

  void freeAllBlocks() {
    // Release the used blocks
    while(usedBlocks) {
      releaseBlock(usedBlocks, NULL);
    }

    // Release the unused blocks
    while(freeBlocks) {
      assert(freeBlocks->isHead);
      MA::deallocate(freeBlocks->data);
      totalBytes -= freeBlocks->size;
      struct Block *curr = freeBlocks;
      freeBlocks = freeBlocks->next;
      blockAllocator.deallocate(curr);
    }
  }

public:
  static inline DynamicPoolAllocator &getInstance() {
    static DynamicPoolAllocator instance;
    return instance;
  }

  DynamicPoolAllocator(const std::size_t _minBytes = (1 << 8))
    : blockAllocator(),
      usedBlocks(NULL),
      freeBlocks(NULL),
      totalBytes(0),
      allocBytes(0),
      minBytes(_minBytes) { }

  ~DynamicPoolAllocator() { freeAllBlocks(); }

  void *allocate(std::size_t size) {
    struct Block *best, *prev;
    findUsableBlock(best, prev, size);

    // Allocate a block if needed
    if (!best) allocateBlock(best, prev, size);
    assert(best);

    // Split the free block
    splitBlock(best, prev, size);

    // Push node to the list of used nodes
    best->next = usedBlocks;
    usedBlocks = best;

    // Increment the allocated size
    allocBytes += size;

    // Return the new pointer
    return usedBlocks->data;
  }

  void deallocate(void *ptr) {
    assert(ptr);

    // Find the associated block
    struct Block *curr = usedBlocks, *prev = NULL;
    for ( ; curr && curr->data != ptr; curr = curr->next ) {
      prev = curr;
    }
    if (!curr) return;

    // Remove from allocBytes
    allocBytes -= curr->size;

    // Release it
    releaseBlock(curr, prev);
  }

  std::size_t allocatedSize() const { return allocBytes; }

  std::size_t totalSize() const {
    return totalBytes + blockAllocator.totalSize();
  }

  std::size_t numFreeBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = freeBlocks; temp; temp = temp->next) nb++;
    return nb;
  }

  std::size_t numUsedBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = usedBlocks; temp; temp = temp->next) nb++;
    return nb;
  }

};


#endif
