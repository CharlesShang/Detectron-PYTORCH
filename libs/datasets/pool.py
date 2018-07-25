import threading
import collections

from multiprocessing.pool import RUN, job_counter, TimeoutError, mapstar
from multiprocessing.queues import Empty, Full
from torch.multiprocessing import Queue
from torch.multiprocessing.pool import Pool


#
# Class whose instances are returned by `Pool.imap()`
#

class sIMapIterator(object):

    def __init__(self, cache, maxsize):
        self._cond = threading.Condition(threading.Lock())

        self._empty_sema = threading.Semaphore(maxsize)
        # self._full_sema = threading.Semaphore(0)

        self._job = job_counter.next()
        self._cache = cache
        # self._items = collections.deque()
        self._items = Queue(maxsize)
        # print self._items.maxsize

        self._index = 0
        # self._put_index = 0
        # self._get_index = 0
        self._length = None
        #
        # self._get_lock = threading.Lock()
        # self._put_lock = threading.Lock()

        self._unsorted = {}
        cache[self._job] = self

    def __iter__(self):
        return self

    def next(self, timeout=None):
        # with self._get_lock:
        #     if self._get_index == self._length:
        #         raise StopIteration
        #     item = self._items.get(timeout=timeout)
        #     self._get_index += 1
        #
        #     success, value = item
        #     if success:
        #         return value
        #     raise value

        self._cond.acquire()
        try:
            try:
                item = self._items.get_nowait()
                self._empty_sema.release()
            except Empty:
                if self._index == self._length:
                    raise StopIteration
                self._cond.wait(timeout)
                try:
                    item = self._items.get(timeout=timeout)
                    self._empty_sema.release()
                except Empty:
                    if self._index == self._length:
                        raise StopIteration
                    raise TimeoutError
        finally:
            self._cond.release()

        success, value = item
        if success:
            return value
        raise value

    __next__ = next                    # XXX

    def _set(self, i, obj):
        # with self._put_lock:
        #     if self._put_index != i:
        #         self._unsorted[i] = obj
        #     else:
        #         self._items.put(obj)
        #         self._put_index += 1
        #         while self._put_index in self._unsorted:
        #             obj = self._unsorted.pop(self._put_index)
        #             self._items.put(obj)
        #             self._put_index += 1
        #
        #     if self._put_index == self._length:
        #         del self._cache[self._job]

        self._empty_sema.acquire()
        self._cond.acquire()
        try:
            if self._index == i:
                self._items.put_nowait(obj)
                self._index += 1
                while self._index in self._unsorted:
                    obj = self._unsorted.pop(self._index)
                    self._items.put_nowait(obj)
                    self._index += 1
                self._cond.notify()
            else:
                self._unsorted[i] = obj

            if self._index == self._length:
                del self._cache[self._job]
        finally:
            self._cond.release()

    def _set_length(self, length):
        #
        # with self._put_lock as pl, self._get_lock as gl:
        #     self._length = length
        #     if self._put_index == self._length:
        #         del self._cache[self._job]

        self._cond.acquire()
        try:
            self._length = length
            if self._index == self._length:
                self._cond.notify()
                del self._cache[self._job]
        finally:
            self._cond.release()


#
# Class whose instances are returned by `Pool.imap_unordered()`
#
class sIMapUnorderedIterator(sIMapIterator):

    def _set(self, i, obj):
    #     with self._put_lock:
    #         self._items.put(obj)
    #         self._put_index += 1
    #
    #         if self._put_lock == self._length:
    #             del self._cache[self._job]

        self._empty_sema.acquire()
        self._cond.acquire()
        try:
            self._items.put_nowait(obj)
            self._index += 1
            self._cond.notify()
            if self._index == self._length:
                del self._cache[self._job]
        finally:
            self._cond.release()


class sPool(Pool):

    def imap(self, func, iterable, chunksize=1, maxsize=0):
        '''
        Equivalent of `itertools.imap()` -- can be MUCH slower than `Pool.map()`
        '''
        assert self._state == RUN
        if chunksize == 1:
            result = sIMapIterator(self._cache, maxsize=maxsize)
            self._taskqueue.put((((result._job, i, func, (x,), {})
                         for i, x in enumerate(iterable)), result._set_length))
            return result
        else:
            assert chunksize > 1
            task_batches = Pool._get_tasks(func, iterable, chunksize)
            result = sIMapIterator(self._cache, maxsize=maxsize)
            self._taskqueue.put((((result._job, i, mapstar, (x,), {})
                     for i, x in enumerate(task_batches)), result._set_length))
            return (item for chunk in result for item in chunk)

    def imap_unordered(self, func, iterable, chunksize=1, maxsize=0):
        '''
        Like `imap()` method but ordering of results is arbitrary
        '''
        assert self._state == RUN
        if chunksize == 1:
            result = sIMapUnorderedIterator(self._cache, maxsize=maxsize)
            self._taskqueue.put((((result._job, i, func, (x,), {})
                         for i, x in enumerate(iterable)), result._set_length))
            return result
        else:
            assert chunksize > 1
            task_batches = Pool._get_tasks(func, iterable, chunksize)
            result = sIMapUnorderedIterator(self._cache, maxsize=maxsize)
            self._taskqueue.put(
                (((result._job, i, mapstar, (x,), {}) for i, x in enumerate(task_batches)), result._set_length)
            )
            return (item for chunk in result for item in chunk)